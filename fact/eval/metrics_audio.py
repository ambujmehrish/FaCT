"""Waveform-level reconstruction metrics with reference implementations.

PESQ (ITU-T P.862, via the `pesq` package) and STOI (via `pystoi`) are the
two externally-verifiable numbers - both are the exact implementations the
codec literature reports, so our tables are comparable to published ones.
Both operate at 16 kHz mono on CPU (they are C routines; cheap relative to
model inference, which stays on GPU).

Heavier model-based metrics (WER via Whisper, speaker-SIM via WavLM-ECAPA,
UTMOSv2, emotion2vec) are pluggable callables - see scripts/eval_baselines.py
--wer flag for the Whisper hook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

METRIC_SR = 16_000


def resample(wav: torch.Tensor, sr_from: int, sr_to: int) -> torch.Tensor:
    if sr_from == sr_to:
        return wav
    import torchaudio
    return torchaudio.functional.resample(wav, sr_from, sr_to)


def _align(ref: np.ndarray, hyp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(ref), len(hyp))
    return ref[:n], hyp[:n]


def pesq_wb(ref: torch.Tensor, hyp: torch.Tensor, sr: int) -> float:
    """PESQ wideband (P.862.2) at 16 kHz. ref/hyp: (N,) tensors at `sr`."""
    from pesq import pesq
    r = resample(ref.detach().cpu().float(), sr, METRIC_SR).numpy()
    h = resample(hyp.detach().cpu().float(), sr, METRIC_SR).numpy()
    r, h = _align(r, h)
    return float(pesq(METRIC_SR, r, h, "wb"))


def stoi(ref: torch.Tensor, hyp: torch.Tensor, sr: int) -> float:
    from pystoi import stoi as _stoi
    r = resample(ref.detach().cpu().float(), sr, METRIC_SR).numpy()
    h = resample(hyp.detach().cpu().float(), sr, METRIC_SR).numpy()
    r, h = _align(r, h)
    return float(_stoi(r, h, METRIC_SR, extended=False))


def si_sdr(ref: torch.Tensor, hyp: torch.Tensor) -> float:
    """Scale-invariant SDR in dB (time-domain; sensitive to phase/latency -
    report alongside PESQ, never alone, for generative decoders)."""
    r = ref.detach().cpu().float()
    h = hyp.detach().cpu().float()
    n = min(r.shape[-1], h.shape[-1])
    r, h = r[..., :n], h[..., :n]
    r = r - r.mean()
    h = h - h.mean()
    proj = (torch.dot(h, r) / (torch.dot(r, r) + 1e-8)) * r
    noise = h - proj
    return float(10 * torch.log10(proj.pow(2).sum() / (noise.pow(2).sum() + 1e-8)))


@dataclass
class UtteranceMetrics:
    pesq_wb: Optional[float] = None
    stoi: Optional[float] = None
    si_sdr: Optional[float] = None
    mel_l1: Optional[float] = None
    f0_rmse_hz: Optional[float] = None
    voicing_f1: Optional[float] = None
    energy_corr: Optional[float] = None
    extras: dict[str, float] = field(default_factory=dict)


def aggregate(items: list[UtteranceMetrics]) -> dict[str, dict[str, float]]:
    """Mean/std/count per metric, skipping Nones and failures."""
    out: dict[str, dict[str, float]] = {}
    keys = ["pesq_wb", "stoi", "si_sdr", "mel_l1", "f0_rmse_hz",
            "voicing_f1", "energy_corr"]
    extra_keys = sorted({k for it in items for k in it.extras})
    for key in keys + extra_keys:
        vals = []
        for it in items:
            v = it.extras.get(key) if key in extra_keys else getattr(it, key)
            if v is not None and np.isfinite(v):
                vals.append(v)
        if vals:
            out[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
            }
    return out
