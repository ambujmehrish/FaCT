"""Prosody supervision targets: quantized log-F0 + frame energy.

All targets are cheap and automatically extractable. F0 comes from pyworld
(optional dependency) at preprocessing time; energy is derived from the mel
directly. Targets are pooled to the 12.5 Hz token rate.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..config import AudioConfig, ProsodyConfig


def quantize_f0(f0_hz: torch.Tensor, cfg: ProsodyConfig) -> torch.Tensor:
    """Map F0 (Hz, 0 = unvoiced) to class ids.

    Voiced frames -> log-spaced bins [0, f0_bins); unvoiced -> class f0_bins.
    """
    voiced = f0_hz > 0
    log_f0 = torch.log(f0_hz.clamp_min(1.0))
    lo, hi = math.log(cfg.f0_min), math.log(cfg.f0_max)
    bins = ((log_f0 - lo) / (hi - lo) * cfg.f0_bins).long().clamp(0, cfg.f0_bins - 1)
    return torch.where(voiced, bins, torch.full_like(bins, cfg.f0_bins))


def pool_to_token_rate(x: torch.Tensor, frame_stack: int,
                       reduce: str = "mean") -> torch.Tensor:
    """Pool (B, T_mel) frame-rate targets to (B, T_mel // frame_stack)."""
    b, t = x.shape
    t_trim = (t // frame_stack) * frame_stack
    x = x[:, :t_trim].reshape(b, t_trim // frame_stack, frame_stack)
    if reduce == "mean":
        return x.float().mean(dim=-1)
    if reduce == "median":
        return x.float().median(dim=-1).values
    raise ValueError(reduce)


def energy_from_mel(log_mel: torch.Tensor) -> torch.Tensor:
    """Frame energy proxy: mean log-mel per frame, z-normalized per utterance."""
    e = log_mel.mean(dim=-1)  # (B, T)
    mu = e.mean(dim=1, keepdim=True)
    sd = e.std(dim=1, keepdim=True).clamp_min(1e-5)
    return (e - mu) / sd


def extract_f0_pyworld(wav: np.ndarray, sample_rate: int,
                       hop_length: int, f0_min: float, f0_max: float) -> np.ndarray:
    """F0 track (Hz, 0 = unvoiced) aligned to the mel hop. Requires pyworld."""
    import pyworld  # optional dependency

    wav64 = wav.astype(np.float64)
    frame_period_ms = 1000.0 * hop_length / sample_rate
    f0, t = pyworld.dio(wav64, sample_rate, f0_floor=f0_min, f0_ceil=f0_max,
                        frame_period=frame_period_ms)
    return pyworld.stonemask(wav64, f0, t, sample_rate)


def extract_f0_autocorr(wav: np.ndarray, sample_rate: int, hop_length: int,
                        f0_min: float, f0_max: float,
                        voicing_threshold: float = 0.5) -> np.ndarray:
    """Dependency-free F0 fallback: FFT autocorrelation peak picking.

    Coarser than pyworld (no refinement, simple voicing decision) but keeps
    the pipeline runnable without compiled extras. Prefer pyworld for paper
    runs.
    """
    frame_length = min(int(3 * sample_rate / f0_min), 2048)
    pad = frame_length // 2
    x = np.pad(wav.astype(np.float64), (pad, pad))
    n_frames = len(wav) // hop_length + 1
    idx = np.arange(frame_length)[None, :] + hop_length * np.arange(n_frames)[:, None]
    idx = np.minimum(idx, len(x) - 1)
    frames = x[idx]
    frames = frames - frames.mean(axis=1, keepdims=True)

    n_fft = 2 * frame_length
    spec = np.fft.rfft(frames, n=n_fft)
    acf = np.fft.irfft(spec * np.conj(spec), n=n_fft)[:, :frame_length]
    r0 = np.maximum(acf[:, 0], 1e-10)
    acf_norm = acf / r0[:, None]

    lag_min = max(2, int(sample_rate / f0_max))
    lag_max = min(frame_length - 1, int(sample_rate / f0_min))
    window = acf_norm[:, lag_min : lag_max + 1]
    best = window.argmax(axis=1)
    peak = window[np.arange(n_frames), best]
    lag = best + lag_min
    f0 = sample_rate / lag.astype(np.float64)

    rms = np.sqrt((frames ** 2).mean(axis=1))
    energy_floor = 0.05 * max(np.sqrt((wav.astype(np.float64) ** 2).mean()), 1e-8)
    voiced = (peak > voicing_threshold) & (rms > energy_floor)
    return np.where(voiced, f0, 0.0)


def extract_f0(wav: np.ndarray, sample_rate: int, hop_length: int,
               f0_min: float, f0_max: float) -> np.ndarray:
    """pyworld when available, autocorrelation fallback otherwise."""
    try:
        return extract_f0_pyworld(wav, sample_rate, hop_length, f0_min, f0_max)
    except ImportError:
        global _WARNED_FALLBACK
        if not _WARNED_FALLBACK:
            import warnings
            warnings.warn("pyworld not installed; using coarse autocorrelation F0. "
                          "Install the [audio] extra for paper runs.")
            _WARNED_FALLBACK = True
        return extract_f0_autocorr(wav, sample_rate, hop_length, f0_min, f0_max)


_WARNED_FALLBACK = False


def prosody_targets_from_mel_and_f0(
    log_mel: torch.Tensor, f0_hz: torch.Tensor,
    prosody_cfg: ProsodyConfig, frame_stack: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, T_mel, n_mels), (B, T_mel) -> token-rate (f0_bins, energy)."""
    t = min(log_mel.shape[1], f0_hz.shape[1])
    f0_tok = pool_to_token_rate(f0_hz[:, :t], frame_stack, reduce="median")
    f0_bins = quantize_f0(f0_tok, prosody_cfg)
    energy = pool_to_token_rate(energy_from_mel(log_mel[:, :t]), frame_stack)
    return f0_bins, energy
