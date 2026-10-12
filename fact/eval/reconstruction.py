"""Modelability probes #4/#5: fidelity ceiling + streaming fitness.

Implemented here (dependency-free, mel-domain):
- mel L1/L2 distance
- F0 RMSE (voiced frames) and voicing F1, from provided F0 tracks
- energy correlation
- chunk-boundary discontinuity (MagpieTTS-LF-style delta-F0 / delta-energy at
  block edges vs elsewhere)

External-model metrics (UTMOS, PESQ, ASR-WER, speaker-SIM, emotion-SIM)
are intentionally hooks: pass any callable wav->score; this module handles
aggregation only, so the harness has no heavyweight dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch


@dataclass
class ReconstructionMetrics:
    mel_l1: float
    mel_l2: float
    energy_corr: float
    f0_rmse_hz: Optional[float] = None
    voicing_f1: Optional[float] = None
    external: dict[str, float] = field(default_factory=dict)


def mel_distance(ref: torch.Tensor, hyp: torch.Tensor) -> tuple[float, float]:
    t = min(ref.shape[-2], hyp.shape[-2])
    d = ref[..., :t, :] - hyp[..., :t, :]
    return d.abs().mean().item(), d.pow(2).mean().sqrt().item()


def energy_correlation(ref: torch.Tensor, hyp: torch.Tensor) -> float:
    """Pearson correlation of frame energies (mean log-mel)."""
    t = min(ref.shape[-2], hyp.shape[-2])
    a = ref[..., :t, :].mean(-1).flatten()
    b = hyp[..., :t, :].mean(-1).flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-8)
    return float((a @ b) / denom)

def f0_metrics(ref_f0: torch.Tensor, hyp_f0: torch.Tensor) -> tuple[float, float]:
    """(RMSE over jointly-voiced frames in Hz, voicing F1)."""
    t = min(ref_f0.shape[-1], hyp_f0.shape[-1])
    ref_f0, hyp_f0 = ref_f0[..., :t], hyp_f0[..., :t]
    rv, hv = ref_f0 > 0, hyp_f0 > 0
    both = rv & hv
    rmse = float(((ref_f0[both] - hyp_f0[both]) ** 2).mean().sqrt()) if both.any() else float("nan")
    tp = (rv & hv).sum().item()
    prec = tp / max(1, hv.sum().item())
    rec = tp / max(1, rv.sum().item())
    f1 = 2 * prec * rec / max(1e-8, prec + rec)
    return rmse, f1


def reconstruction_metrics(
    ref_mel: torch.Tensor, hyp_mel: torch.Tensor,
    ref_f0: Optional[torch.Tensor] = None, hyp_f0: Optional[torch.Tensor] = None,
    external_metrics: Optional[dict[str, Callable[[], float]]] = None,
) -> ReconstructionMetrics:
    l1, l2 = mel_distance(ref_mel, hyp_mel)
    out = ReconstructionMetrics(
        mel_l1=l1, mel_l2=l2, energy_corr=energy_correlation(ref_mel, hyp_mel)
    )
    if ref_f0 is not None and hyp_f0 is not None:
        out.f0_rmse_hz, out.voicing_f1 = f0_metrics(ref_f0, hyp_f0)
    if external_metrics:
        out.external = {k: fn() for k, fn in external_metrics.items()}
    return out


def chunk_boundary_discontinuity(mel: torch.Tensor, block_size: int) -> dict[str, float]:
    """Delta-energy at block boundaries vs within blocks.

    A streaming decoder with visible seams shows boundary deltas well above
    the within-block baseline; ratio ~1.0 means seam-free.
    """
    e = mel.mean(dim=-1)  # (B, T) frame energy
    delta = (e[:, 1:] - e[:, :-1]).abs()
    t = delta.shape[1]
    idx = torch.arange(1, t + 1)
    at_boundary = (idx % block_size) == 0
    boundary = delta[:, at_boundary].mean().item() if at_boundary.any() else float("nan")
    within = delta[:, ~at_boundary].mean().item()
    return {
        "boundary_delta": boundary,
        "within_delta": within,
        "ratio": boundary / max(1e-8, within),
    }
