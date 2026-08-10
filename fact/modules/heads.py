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


def ctc_required_length(targets: torch.Tensor,
                        target_lengths: torch.Tensor) -> torch.Tensor:
    """Minimum CTC input length per item: S + adjacent repeated labels
    (repeats need an intervening blank). The naive S-only check undercounts
    on byte text ('ll', 'ss', ...)."""
    if targets.shape[1] < 2:
        return target_lengths
    same = targets[:, 1:] == targets[:, :-1]
    pos = torch.arange(1, targets.shape[1], device=targets.device)
    within = pos.unsqueeze(0) < target_lengths.unsqueeze(1)
    return target_lengths + (same & within).sum(dim=1)


def masked_ctc_loss(logits: torch.Tensor, targets: torch.Tensor,
                    target_lengths: torch.Tensor, input_lengths: torch.Tensor,
                    blank: int) -> torch.Tensor:
    """CTC averaged over items that HAVE a transcript, normalized per label.

    Items with target_length 0 (e.g. audio-only utterances, or crops whose
    transcript was dropped) are excluded - they neither contribute loss nor
    get pushed toward all-blank.
    """
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, V)
    losses = F.ctc_loss(
        log_probs, targets, input_lengths, target_lengths,
        blank=blank, reduction="none", zero_infinity=True,
    )
    valid = target_lengths > 0
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return (losses[valid] / target_lengths[valid].float()).mean()


class CTCHead(nn.Module):
    """Causal transformer + CTC over the quantized content stream.

    Emits `cfg.upsample` logit positions per token frame so byte-level
    targets stay feasible at low token rates (see CTCConfig docstring).
    """

    def __init__(self, in_dim: int, cfg: CTCConfig):
        super().__init__()
        self.cfg = cfg
        self.proj_in = nn.Linear(in_dim, cfg.dim)
        self.transformer = CausalTransformer(cfg.dim, cfg.n_layers, cfg.n_heads)
        self.proj_out = nn.Linear(cfg.dim, cfg.upsample * cfg.vocab_size)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, T * upsample, vocab)."""
        b, t, _ = x.shape
        out = self.proj_out(self.transformer(self.proj_in(x)))
        return out.reshape(b, t * self.cfg.upsample, self.cfg.vocab_size)

    def forward(self, x: torch.Tensor, targets: torch.Tensor,
                target_lengths: torch.Tensor,
                token_lengths: Optional[torch.Tensor] = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, D) content embeddings; targets: (B, S) label ids (0 = blank pad).

        token_lengths: per-item VALID token count (padding excluded); None
        means all T frames are valid.

        Returns (ctc_loss, infeasible_fraction). infeasible counts items -
        among those with transcripts - whose required CTC length (labels +
        repeat blanks) exceeds their valid input positions; those items are
        zeroed by zero_infinity, and this fraction makes that visible. It
        must stay ~0 in real training.
        """
        logits = self.logits(x)
        t_up = logits.shape[1]
        if token_lengths is None:
            input_lengths = torch.full((x.shape[0],), t_up, dtype=torch.long,
                                       device=x.device)
        else:
            input_lengths = (token_lengths * self.cfg.upsample).clamp(max=t_up)
        required = ctc_required_length(targets, target_lengths)
        valid = target_lengths > 0
        infeasible = ((required > input_lengths) & valid).float().sum() \
            / valid.float().sum().clamp_min(1)
        loss = masked_ctc_loss(logits, targets, target_lengths, input_lengths,
                               self.cfg.blank_id)
        return loss, infeasible

    @torch.no_grad()
    def greedy_decode(self, x: torch.Tensor) -> list[list[int]]:
        """Collapsed label ids per item, in the SHIFTED space (byte value + 1,
        0 = blank; see preprocess.encode_text_bytes). Use `ids_to_text` to
        recover strings."""
        pred = self.logits(x).argmax(dim=-1)  # (B, T * upsample)
        out = []
        for seq in pred.tolist():
            collapsed, prev = [], None
            for tok in seq:
                if tok != prev and tok != self.cfg.blank_id:
                    collapsed.append(tok)
                prev = tok
            out.append(collapsed)
        return out

    @staticmethod
    def ids_to_text(ids: list[int]) -> str:
        """Invert encode_text_bytes: shifted label ids -> UTF-8 string."""
        return bytes(i - 1 for i in ids if i > 0).decode("utf-8", errors="replace")


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
                energy: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, D); f0_bins: (B, T) long in [0, f0_bins], padding = -100
        (ignored); energy: (B, T); mask: (B, T) bool, True = valid frame."""
        h = self.net(x)
        f0_loss = F.cross_entropy(
            self.f0_head(h).transpose(1, 2), f0_bins,
            reduction="mean", ignore_index=-100,
        )
        err = (self.energy_head(h).squeeze(-1) - energy).pow(2)
        if mask is None:
            energy_loss = err.mean()
        else:
            m = mask.float()
            energy_loss = (err * m).sum() / m.sum().clamp_min(1)
        return f0_loss, energy_loss


class LeakageProbes(nn.Module):
    """Adversarial probes with gradient reversal.

    The probes themselves learn to predict the forbidden attribute as well
    as possible; the reversed gradient teaches the encoder to remove it.
    Probe accuracies (measured with detached inputs at eval time) populate
    the paper's leakage matrix.
    """

    def __init__(self, in_dim: int, cfg: LeakageConfig,
                 f0_classes: int, text_vocab: int, text_upsample: int = 2):
        super().__init__()
        self.cfg = cfg
        self.text_vocab = text_vocab
        self.text_upsample = text_upsample  # match CTCConfig.upsample (feasibility)
        h = cfg.probe_dim
        self.content_to_f0 = nn.Sequential(
            nn.Linear(in_dim, h), nn.SiLU(), nn.Linear(h, f0_classes)
        )
        self.prosody_to_text = nn.Sequential(
            nn.Linear(in_dim, h), nn.SiLU(), nn.Linear(h, text_upsample * text_vocab)
        )
        self.text_blank = 0

    def forward(self, content_emb: torch.Tensor, prosody_emb: torch.Tensor,
                f0_bins: torch.Tensor, text_targets: torch.Tensor,
                target_lengths: torch.Tensor,
                token_lengths: Optional[torch.Tensor] = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """f0_bins padding must be -100 (ignored); token_lengths as in CTCHead."""
        lam = self.cfg.grl_lambda
        # Content must NOT predict F0.
        c_rev = grad_reverse(content_emb, lam)
        c2f_loss = F.cross_entropy(
            self.content_to_f0(c_rev).transpose(1, 2), f0_bins,
            reduction="mean", ignore_index=-100,
        )
        # Prosody must NOT win at CTC (upsampled like the main CTC head so
        # the probe is feasible wherever the main head is).
        p_rev = grad_reverse(prosody_emb, lam)
        b, t, _ = p_rev.shape
        logits = self.prosody_to_text(p_rev).reshape(
            b, t * self.text_upsample, self.text_vocab
        )
        if token_lengths is None:
            input_lengths = torch.full((b,), logits.shape[1], dtype=torch.long,
                                       device=p_rev.device)
        else:
            input_lengths = (token_lengths * self.text_upsample).clamp(
                max=logits.shape[1]
            )
        p2t_loss = masked_ctc_loss(logits, text_targets, target_lengths,
                                   input_lengths, self.text_blank)
        return c2f_loss, p2t_loss
