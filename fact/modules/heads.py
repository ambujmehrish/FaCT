"""Supervision and leakage heads.

- CTCHead: 4-layer causal transformer predicting text from *quantized*
  content embeddings with CTC loss (SiTok's mechanism - no SSL teacher at
  inference). Applied only to the content stream: this asymmetry is what
  creates the factorization pressure.
- ProsodyPredictor: supervises the prosody stream with cheap targets
  (quantized log-F0 classes + frame energy regression).
- Gradient-reversal leakage probes (DSA-style):
  content -> F0 (content must not encode prosody),
  prosody -> text (prosody must not encode content).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..config import CTCConfig, LeakageConfig, ProsodyConfig
from .transformer import CausalTransformer


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.lam = lam
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return -ctx.lam * grad, None


def grad_reverse(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, lam)


class CTCHead(nn.Module):
    """Causal transformer + CTC over the quantized content stream."""

    def __init__(self, in_dim: int, cfg: CTCConfig):
        super().__init__()
        self.cfg = cfg
        self.proj_in = nn.Linear(in_dim, cfg.dim)
        self.transformer = CausalTransformer(cfg.dim, cfg.n_layers, cfg.n_heads)
        self.proj_out = nn.Linear(cfg.dim, cfg.vocab_size)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_out(self.transformer(self.proj_in(x)))

    def forward(self, x: torch.Tensor, targets: torch.Tensor,
                input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) content embeddings; targets: (B, S) label ids (0 = blank pad).

        Returns the CTC loss. Batch items whose targets are longer than their
        token sequence contribute zero (CTC would be infeasible).
        """
        logits = self.logits(x)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, V)
        loss = F.ctc_loss(
            log_probs, targets, input_lengths, target_lengths,
            blank=self.cfg.blank_id, reduction="mean", zero_infinity=True,
        )
        return loss

    @torch.no_grad()
    def greedy_decode(self, x: torch.Tensor) -> list[list[int]]:
        pred = self.logits(x).argmax(dim=-1)  # (B, T)
        out = []
        for seq in pred.tolist():
            collapsed, prev = [], None
            for tok in seq:
                if tok != prev and tok != self.cfg.blank_id:
                    collapsed.append(tok)
                prev = tok
            out.append(collapsed)
        return out


class ProsodyPredictor(nn.Module):
    """Direct supervision of the prosody stream: F0 classes + energy."""

    def __init__(self, in_dim: int, cfg: ProsodyConfig):
        super().__init__()
        self.cfg = cfg
        hidden = cfg.predictor_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        # +1 class for unvoiced frames.
        self.f0_head = nn.Linear(hidden, cfg.f0_bins + 1)
        self.energy_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, f0_bins: torch.Tensor,
                energy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, D); f0_bins: (B, T) long in [0, f0_bins]; energy: (B, T)."""
        h = self.net(x)
        f0_loss = F.cross_entropy(
            self.f0_head(h).transpose(1, 2), f0_bins, reduction="mean"
        )
        energy_loss = F.mse_loss(self.energy_head(h).squeeze(-1), energy)
        return f0_loss, energy_loss


class LeakageProbes(nn.Module):
    """Adversarial probes with gradient reversal.

    The probes themselves learn to predict the forbidden attribute as well
    as possible; the reversed gradient teaches the encoder to remove it.
    Probe accuracies (measured with detached inputs at eval time) populate
    the paper's leakage matrix.
    """

    def __init__(self, in_dim: int, cfg: LeakageConfig,
                 f0_classes: int, text_vocab: int):
        super().__init__()
        self.cfg = cfg
        h = cfg.probe_dim
        self.content_to_f0 = nn.Sequential(
            nn.Linear(in_dim, h), nn.SiLU(), nn.Linear(h, f0_classes)
        )
        self.prosody_to_text = nn.Sequential(
            nn.Linear(in_dim, h), nn.SiLU(), nn.Linear(h, text_vocab)
        )
        self.text_blank = 0

    def forward(self, content_emb: torch.Tensor, prosody_emb: torch.Tensor,
                f0_bins: torch.Tensor,
                text_targets: torch.Tensor, input_lengths: torch.Tensor,
                target_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lam = self.cfg.grl_lambda
        # Content must NOT predict F0.
        c_rev = grad_reverse(content_emb, lam)
        c2f_loss = F.cross_entropy(
            self.content_to_f0(c_rev).transpose(1, 2), f0_bins, reduction="mean"
        )
        # Prosody must NOT win at CTC.
        p_rev = grad_reverse(prosody_emb, lam)
        log_probs = F.log_softmax(self.prosody_to_text(p_rev), dim=-1).transpose(0, 1)
        p2t_loss = F.ctc_loss(
            log_probs, text_targets, input_lengths, target_lengths,
            blank=self.text_blank, reduction="mean", zero_infinity=True,
        )
        return c2f_loss, p2t_loss
