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

    def encode_mel(self, mel: torch.Tensor) -> BottleneckOutput:
        """mel: raw log-mel (normalization is internal)."""
        return self.bottleneck(self.encoder(self._norm_mel(mel)))

    @torch.no_grad()
    def tokenize(self, wav: Optional[torch.Tensor] = None,
                 mel: Optional[torch.Tensor] = None) -> TokenizerOutput:
        if mel is None:
            if wav is None:
                raise ValueError("Provide wav or mel")
            mel = self.mel_transform(wav)
        out = self.encode_mel(mel)
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
                   nfe: int = 2, cfg_scale: float = 1.0) -> torch.Tensor:
        """Discrete view -> mel. The pure index path an LM stack would use."""
        cond = self.bottleneck.embed_indices(content_indices, prosody_indices, residual)
        spk = self.speaker_encoder(self._norm_mel(ref_mel))
        return self._denorm_mel(self.decoder.sample(cond, spk, nfe=nfe, cfg_scale=cfg_scale))

    @torch.no_grad()
    def reconstruct(self, mel: torch.Tensor, ref_mel: Optional[torch.Tensor] = None,
                    nfe: int = 2) -> torch.Tensor:
        """Continuous view round-trip (the fidelity ceiling measurement)."""
        out = self.encode_mel(mel)
        spk = self.speaker_encoder(self._norm_mel(ref_mel if ref_mel is not None else mel))
        return self._denorm_mel(self.decoder.sample(out.decoder_condition(), spk, nfe=nfe))

    # --------------------------------------------------------------- training

    def training_step(self, batch: Batch, stage: str = "a") -> dict[str, torch.Tensor]:
        cfg = self.cfg
        out = self.encode_mel(batch.mel)
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

        spk = self.speaker_encoder(
            self._norm_mel(batch.ref_mel if batch.ref_mel is not None else batch.mel),
            lengths=batch.mel_lengths,
        )
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
