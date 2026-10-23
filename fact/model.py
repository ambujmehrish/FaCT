"""FaCT: Factorized Causal Tokenizer - top-level model.

Pipeline (all causal):
  wav -> log-mel @ 50 Hz -> CausalEncoder -> 12.5 Hz frames
      -> FactorizedBottleneck (content FSQ + prosody FSQ + continuous residual)
      -> FlowMatchingDecoder (block-causal, shortcut-distillable) -> mel

Training losses (stage A):
  flow matching + CTC-on-content + prosody supervision (F0/energy)
  + GRL leakage penalties + residual KL.

Public interfaces:
  tokenize()        wav/mel -> discrete indices + continuous views
  detokenize()      indices (+ optional residual) + speaker ref -> mel
  training_step()   full stage-A loss dict
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .config import FaCTConfig
from .modules.bottleneck import BottleneckOutput, FactorizedBottleneck
from .modules.decoder import FlowMatchingDecoder
from .modules.encoder import CausalEncoder
from .modules.heads import CTCHead, LeakageProbes, ProsodyPredictor
from .modules.mel import LogMelSpectrogram
from .modules.speaker import SpeakerEncoder


@dataclass
class Batch:
    """One training batch. Prosody targets are at the 12.5 Hz token rate.

    Variable-length batches are PADDED (never truncated - truncation would
    desynchronize audio from its transcript): mel padding = audio.mel_mean,
    f0_bins padding = -100 (ignored by losses), energy padding = 0.
    mel_lengths carries the per-item valid frame counts; every loss masks
    on it.
    """

    mel: torch.Tensor            # (B, T_mel, n_mels), padded
    text: torch.Tensor           # (B, S) label ids, 0 = blank/pad
    text_lengths: torch.Tensor   # (B,); 0 = no transcript (excluded from CTC)
    f0_bins: torch.Tensor        # (B, T_tok) long in [0, f0_bins]; pad = -100
    energy: torch.Tensor         # (B, T_tok) float; pad = 0
    ref_mel: Optional[torch.Tensor] = None  # (B, T_ref, n_mels); defaults to mel
    mel_lengths: Optional[torch.Tensor] = None  # (B,) valid mel frames; None = all
    # SSL-input runs only (encoder.ssl_model): 16 kHz audio of exactly
    # (mel_lengths - 1) * 320 samples per item, zero-padded, and the frozen
    # model's features (filled on the GPU by the trainer, fact/modules/ssl.py).
    wav16: Optional[torch.Tensor] = None        # (B, N)
    wav_lengths: Optional[torch.Tensor] = None  # (B,)
    ssl: Optional[torch.Tensor] = None          # (B, T_mel, ssl_dim)
    # Distillation target (distill.weight > 0): frozen teacher features of
    # the same audio, aligned with mel like `ssl` (filled by the trainer).
    teacher: Optional[torch.Tensor] = None      # (B, T_mel, distill.dim)


@dataclass
class TokenizerOutput:
    content_indices: Optional[torch.Tensor]     # None for the "vae" variant
    prosody_indices: Optional[torch.Tensor]     # None unless "factorized"
    content_continuous: Optional[torch.Tensor]
    prosody_continuous: Optional[torch.Tensor]
    residual: Optional[torch.Tensor]
    bottleneck: BottleneckOutput

    def index_streams(self) -> list[tuple[str, torch.Tensor]]:
        return self.bottleneck.index_streams()


class FaCT(nn.Module):
    def __init__(self, cfg: FaCTConfig):
        super().__init__()
        self.cfg = cfg
        self.mel_transform = LogMelSpectrogram(cfg.audio)
        self.encoder = CausalEncoder(cfg.audio, cfg.encoder)
        self.bottleneck = FactorizedBottleneck(cfg.encoder.dim, cfg.bottleneck)
        self.ctc_head = CTCHead(cfg.encoder.dim, cfg.ctc)
        # Prosody supervision and leakage penalties only exist when there is
        # a prosody stream to shape ("factorized" and its "rvq" quantizer-swap).
        self.prosody_predictor = None
        self.leakage = None
        if cfg.bottleneck.variant in ("factorized", "rvq"):
            self.prosody_predictor = ProsodyPredictor(cfg.encoder.dim, cfg.prosody)
            self.leakage = LeakageProbes(
                cfg.encoder.dim, cfg.leakage,
                f0_classes=cfg.prosody.f0_bins + 1,
                text_vocab=cfg.ctc.vocab_size,
                text_upsample=cfg.ctc.upsample,
            )
        self.distill_head = None
        if cfg.distill.weight > 0:
            if cfg.bottleneck.variant not in ("factorized", "rvq", "single_fsq"):
                raise ValueError("distill needs a content token stream")
            self.distill_head = nn.Linear(cfg.encoder.dim, cfg.distill.dim)
        self.speaker_encoder = SpeakerEncoder(cfg.audio, cfg.speaker)
        self.decoder = FlowMatchingDecoder(
            cfg.audio, cfg.decoder,
            cond_dim=cfg.encoder.dim,
            spk_dim=cfg.speaker.dim,
            frame_stack=cfg.encoder.frame_stack,
        )

    def forward(self, batch: Batch, stage: str = "a") -> dict[str, torch.Tensor]:
        """DDP entry point: forward == training_step so gradient hooks fire."""
        return self.training_step(batch, stage=stage)

    # ------------------------------------------------------------- interfaces

    def _norm_mel(self, mel: torch.Tensor) -> torch.Tensor:
        return (mel - self.cfg.audio.mel_mean) / self.cfg.audio.mel_std

    def _denorm_mel(self, mel: torch.Tensor) -> torch.Tensor:
        return mel * self.cfg.audio.mel_std + self.cfg.audio.mel_mean

    def encode_mel(self, mel: torch.Tensor, ssl: Optional[torch.Tensor] = None) -> BottleneckOutput:
        """mel: raw log-mel (normalization is internal); ssl: frozen SSL
        features aligned with mel, required iff encoder.ssl_model is set."""
        return self.bottleneck(self.encoder(self._norm_mel(mel), ssl))

    @torch.no_grad()
    def tokenize(self, wav: Optional[torch.Tensor] = None,
                 mel: Optional[torch.Tensor] = None,
                 ssl: Optional[torch.Tensor] = None) -> TokenizerOutput:
        if mel is None:
            if wav is None:
                raise ValueError("Provide wav or mel")
            mel = self.mel_transform(wav)
        out = self.encode_mel(mel, ssl)
        return TokenizerOutput(
            content_indices=out.content_indices,
            prosody_indices=out.prosody_indices,
            content_continuous=out.content_continuous,
            prosody_continuous=out.prosody_continuous,
            residual=out.residual,
            bottleneck=out,
        )

    @torch.no_grad()
    def detokenize(self, content_indices: torch.Tensor,
                   prosody_indices: Optional[torch.Tensor],
                   ref_mel: torch.Tensor, residual: Optional[torch.Tensor] = None,
                   nfe: int = 2, cfg_scale: float = 1.0,
                   shortcut: bool = True, temperature: float = 1.0) -> torch.Tensor:
        """Discrete view -> mel. The pure index path an LM stack would use."""
        cond = self.bottleneck.embed_indices(content_indices, prosody_indices, residual)
        spk = self.speaker_encoder(self._norm_mel(ref_mel))
        return self._denorm_mel(self.decoder.sample(cond, spk, nfe=nfe, cfg_scale=cfg_scale,
                                                    temperature=temperature, shortcut=shortcut))

    @torch.no_grad()
    def reconstruct(self, mel: torch.Tensor, ref_mel: Optional[torch.Tensor] = None,
                    nfe: int = 2, shortcut: bool = True, cfg_scale: float = 1.0,
                    temperature: float = 1.0, ssl: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Continuous view round-trip (the fidelity ceiling measurement)."""
        out = self.encode_mel(mel, ssl)
        spk = self.speaker_encoder(self._norm_mel(ref_mel if ref_mel is not None else mel))
        return self._denorm_mel(self.decoder.sample(out.decoder_condition(), spk, nfe=nfe,
                                                    cfg_scale=cfg_scale, temperature=temperature,
                                                    shortcut=shortcut))

    # --------------------------------------------------------------- training

    @staticmethod
    def _disjoint_reference(mel: torch.Tensor, lengths: torch.Tensor):
        """Per item: a random 25-50 % prefix or suffix of its valid frames as
        the speaker reference. Returns (ref (B, R, n_mels) left-aligned and
        zero-padded, ref lengths (B,), in_ref (B, T) bool: the frames used as
        reference, to be excluded from the reconstruction loss)."""
        b, t, n = mel.shape
        dev = mel.device
        L = lengths.clamp(min=2, max=t)
        frac = 0.25 + 0.25 * torch.rand(b, device=dev)
        r = (L.float() * frac).long().clamp(min=1)
        suffix = torch.rand(b, device=dev) < 0.5
        start = torch.where(suffix, L - r, torch.zeros_like(r))
        pos = torch.arange(t, device=dev)[None]
        in_ref = (pos >= start[:, None]) & (pos < (start + r)[:, None])
        r_max = int(r.max())
        idx = (start[:, None] + torch.arange(r_max, device=dev)[None]).clamp(max=t - 1)
        ref = torch.gather(mel, 1, idx[..., None].expand(-1, -1, n))
        ref = ref * (torch.arange(r_max, device=dev)[None] < r[:, None])[..., None]
        return ref, r, in_ref

    def training_step(self, batch: Batch, stage: str = "a") -> dict[str, torch.Tensor]:
        cfg = self.cfg
        out = self.encode_mel(batch.mel, batch.ssl)
        t_tok = out.primary_emb().shape[1]
        s = cfg.encoder.frame_stack
        b = batch.mel.shape[0]
        device = batch.mel.device

        # Per-item valid lengths -> masks; padded frames never contribute to
        # any loss (see Batch docstring).
        if batch.mel_lengths is not None:
            token_lengths = (batch.mel_lengths // s).clamp(min=1, max=t_tok)
        else:
            token_lengths = torch.full((b,), t_tok, dtype=torch.long, device=device)
        tok_mask = (torch.arange(t_tok, device=device).unsqueeze(0)
                    < token_lengths.unsqueeze(1))
        mel_mask = tok_mask.repeat_interleave(s, dim=1)

        # Align token-rate targets (prosody extraction may differ by a frame).
        f0_bins = batch.f0_bins[:, :t_tok]
        energy = batch.energy[:, :t_tok]

        if cfg.speaker.ref_mode == "disjoint" and batch.ref_mel is None:
            ref, ref_len, in_ref = self._disjoint_reference(batch.mel, token_lengths * s)
            spk = self.speaker_encoder(self._norm_mel(ref), lengths=ref_len)
            mel_mask = mel_mask & ~in_ref[:, : mel_mask.shape[1]]
        elif cfg.speaker.ref_mode in ("same", "disjoint"):
            spk = self.speaker_encoder(
                self._norm_mel(batch.ref_mel if batch.ref_mel is not None else batch.mel),
                lengths=batch.mel_lengths,
            )
        else:
            raise ValueError(f"unknown speaker.ref_mode {cfg.speaker.ref_mode!r}")
        p_drop = cfg.bottleneck.residual_dropout
        if self.training and p_drop > 0 and out.residual_emb is not None \
                and out.content_emb is not None:
            keep = (torch.rand(b, device=device) >= p_drop).to(out.residual_emb.dtype)
            out = dataclasses.replace(out, residual_emb=out.residual_emb * keep[:, None, None])
        cond = out.decoder_condition()
        mel_target = self._norm_mel(batch.mel[:, : t_tok * s])

        if stage == "b":
            recon = self.decoder.shortcut_loss(mel_target, cond, spk, mask=mel_mask)
        else:
            recon = self.decoder.flow_matching_loss(mel_target, cond, spk, mask=mel_mask)

        ctc, ctc_infeasible = self.ctc_head(
            out.primary_emb(), batch.text, batch.text_lengths, token_lengths
        )

        if out.kl_loss.dim() == 0:
            kl = out.kl_loss
        else:
            kl = (out.kl_loss[:, :t_tok] * tok_mask.float()).sum() \
                / tok_mask.float().sum().clamp_min(1)

        total = (
            recon
            + cfg.ctc.weight * ctc
            + cfg.bottleneck.residual_kl_weight * kl
            + out.quantizer_loss  # RVQ commitment; zero for FSQ variants
        )
        logs = {
            "recon": recon.detach(),
            "ctc": ctc.detach(),
            # Fraction of transcript-bearing items whose CTC is infeasible
            # (labels + repeat blanks > valid positions) and silently zeroed.
            # Must stay ~0; if it grows, raise ctc.upsample or check data.
            "ctc_infeasible": ctc_infeasible.detach(),
            "kl": kl.detach(),
        }
        if self.cfg.bottleneck.variant == "rvq":
            logs["commit"] = out.quantizer_loss.detach()

        if self.distill_head is not None:
            distill = self._distill_loss(out.content_emb, batch.teacher, tok_mask, s)
            total = total + cfg.distill.weight * distill
            logs["distill"] = distill.detach()

        # Factorization pressure: only where a prosody stream exists.
        if self.prosody_predictor is not None:
            f0_loss, energy_loss = self.prosody_predictor(
                out.prosody_emb, f0_bins, energy, mask=tok_mask
            )
            c2f, p2t = self.leakage(
                out.content_emb, out.prosody_emb, f0_bins,
                batch.text, batch.text_lengths, token_lengths,
            )
            lk = cfg.leakage
            total = (
                total
                + cfg.prosody.f0_weight * f0_loss
                + cfg.prosody.energy_weight * energy_loss
                + lk.content_to_f0_weight * c2f
                + lk.prosody_to_text_weight * p2t
            )
            logs.update({
                "f0": f0_loss.detach(),
                "energy": energy_loss.detach(),
                "leak_c2f": c2f.detach(),
                "leak_p2t": p2t.detach(),
            })

        logs["loss"] = total
        return logs

    def _distill_loss(self, content_emb: torch.Tensor, teacher: Optional[torch.Tensor],
                      tok_mask: torch.Tensor, s: int) -> torch.Tensor:
        """Mean cosine distance between distill_head(content tokens) and the
        teacher features averaged over each token's s mel frames."""
        if teacher is None:
            raise ValueError("distill.weight > 0 but the batch has no teacher features")
        b, t_tok, _ = content_emb.shape
        if teacher.shape[1] < t_tok * s:
            raise ValueError(f"teacher has {teacher.shape[1]} frames, need {t_tok * s}")
        target = teacher[:, : t_tok * s].reshape(b, t_tok, s, -1).mean(2)
        pred = self.distill_head(content_emb)
        dist = 1 - torch.nn.functional.cosine_similarity(pred.float(), target.float(), dim=-1)
        m = tok_mask.float()
        return (dist * m).sum() / m.sum().clamp_min(1)

    # ------------------------------------------------------------- utilities

    def freeze_tokenizer(self) -> None:
        """Stage B: freeze encoder + bottleneck (SiTok recipe), train decoder only."""
        for module in (self.encoder, self.bottleneck, self.ctc_head,
                       self.prosody_predictor, self.leakage):
            if module is None:
                continue
            for p in module.parameters():
                p.requires_grad_(False)

    def param_counts(self) -> dict[str, int]:
        def count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())
        return {
            "encoder": count(self.encoder),
            "bottleneck": count(self.bottleneck),
            "ctc_head": count(self.ctc_head),
            "decoder": count(self.decoder),
            "speaker": count(self.speaker_encoder),
            "total": count(self),
        }
