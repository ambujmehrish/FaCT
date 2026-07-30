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
    """One training batch. Prosody targets are at the 12.5 Hz token rate."""

    mel: torch.Tensor            # (B, T_mel, n_mels)
    text: torch.Tensor           # (B, S) label ids, 0 = blank/pad
    text_lengths: torch.Tensor   # (B,)
    f0_bins: torch.Tensor        # (B, T_tok) long in [0, f0_bins] (last = unvoiced)
    energy: torch.Tensor         # (B, T_tok) float (normalized log energy)
    ref_mel: Optional[torch.Tensor] = None  # (B, T_ref, n_mels); defaults to mel


@dataclass
class TokenizerOutput:
    content_indices: torch.Tensor
    prosody_indices: torch.Tensor
    content_continuous: torch.Tensor
    prosody_continuous: torch.Tensor
    residual: Optional[torch.Tensor]
    bottleneck: BottleneckOutput


class FaCT(nn.Module):
    def __init__(self, cfg: FaCTConfig):
        super().__init__()
        self.cfg = cfg
        self.mel_transform = LogMelSpectrogram(cfg.audio)
        self.encoder = CausalEncoder(cfg.audio, cfg.encoder)
        self.bottleneck = FactorizedBottleneck(cfg.encoder.dim, cfg.bottleneck)
        self.ctc_head = CTCHead(cfg.encoder.dim, cfg.ctc)
        self.prosody_predictor = ProsodyPredictor(cfg.encoder.dim, cfg.prosody)
        self.leakage = LeakageProbes(
            cfg.encoder.dim, cfg.leakage,
            f0_classes=cfg.prosody.f0_bins + 1,
            text_vocab=cfg.ctc.vocab_size,
        )
        self.speaker_encoder = SpeakerEncoder(cfg.audio, cfg.speaker)
        self.decoder = FlowMatchingDecoder(
            cfg.audio, cfg.decoder,
            cond_dim=cfg.encoder.dim,
            spk_dim=cfg.speaker.dim,
            frame_stack=cfg.encoder.frame_stack,
        )

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
    def detokenize(self, content_indices: torch.Tensor, prosody_indices: torch.Tensor,
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
        t_tok = out.content_emb.shape[1]

        # Align token-rate targets (prosody extraction may differ by a frame).
        f0_bins = batch.f0_bins[:, :t_tok]
        energy = batch.energy[:, :t_tok]

        spk = self.speaker_encoder(self._norm_mel(
            batch.ref_mel if batch.ref_mel is not None else batch.mel
        ))
        cond = out.decoder_condition()
        s = cfg.encoder.frame_stack
        mel_target = self._norm_mel(batch.mel[:, : t_tok * s])

        if stage == "b":
            recon = self.decoder.shortcut_loss(mel_target, cond, spk)
        else:
            recon = self.decoder.flow_matching_loss(mel_target, cond, spk)

        input_lengths = torch.full(
            (out.content_emb.shape[0],), t_tok,
            dtype=torch.long, device=batch.mel.device,
        )
        ctc = self.ctc_head(out.content_emb, batch.text, input_lengths, batch.text_lengths)
        f0_loss, energy_loss = self.prosody_predictor(out.prosody_emb, f0_bins, energy)
        c2f, p2t = self.leakage(
            out.content_emb, out.prosody_emb, f0_bins,
            batch.text, input_lengths, batch.text_lengths,
        )

        lk = cfg.leakage
        total = (
            recon
            + cfg.ctc.weight * ctc
            + cfg.prosody.f0_weight * f0_loss
            + cfg.prosody.energy_weight * energy_loss
            + lk.content_to_f0_weight * c2f
            + lk.prosody_to_text_weight * p2t
            + cfg.bottleneck.residual_kl_weight * out.kl_loss
        )
        return {
            "loss": total,
            "recon": recon.detach(),
            "ctc": ctc.detach(),
            "f0": f0_loss.detach(),
            "energy": energy_loss.detach(),
            "leak_c2f": c2f.detach(),
            "leak_p2t": p2t.detach(),
            "kl": out.kl_loss.detach(),
        }

    # ------------------------------------------------------------- utilities

    def freeze_tokenizer(self) -> None:
        """Stage B: freeze encoder + bottleneck (SiTok recipe), train decoder only."""
        for module in (self.encoder, self.bottleneck, self.ctc_head,
                       self.prosody_predictor, self.leakage):
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
