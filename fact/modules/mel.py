"""Dependency-free log-mel spectrogram (torch.stft + HTK-style filterbank).

Kept minimal so the core repo needs only torch; torchaudio is an optional
extra for I/O-heavy preprocessing.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..config import AudioConfig


def hz_to_mel(f: torch.Tensor | float) -> torch.Tensor | float:
    if isinstance(f, torch.Tensor):
        return 2595.0 * torch.log10(1.0 + f / 700.0)
    return 2595.0 * math.log10(1.0 + f / 700.0)


def mel_to_hz(m: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(n_mels: int, n_fft: int, sample_rate: int,
                   fmin: float, fmax: float) -> torch.Tensor:
    """(n_mels, n_fft // 2 + 1) triangular filterbank."""
    n_freqs = n_fft // 2 + 1
    freqs = torch.linspace(0, sample_rate / 2, n_freqs)
    mel_points = torch.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    fb = torch.zeros(n_mels, n_freqs)
    for i in range(n_mels):
        lo, ctr, hi = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        up = (freqs - lo) / (ctr - lo).clamp_min(1e-8)
        down = (hi - freqs) / (hi - ctr).clamp_min(1e-8)
        fb[i] = torch.clamp(torch.minimum(up, down), min=0.0)
    return fb


class LogMelSpectrogram(nn.Module):
    def __init__(self, cfg: AudioConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("window", torch.hann_window(cfg.win_length), persistent=False)
        self.register_buffer(
            "fb",
            mel_filterbank(cfg.n_mels, cfg.n_fft, cfg.sample_rate, cfg.fmin, cfg.fmax),
            persistent=False,
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, N) -> log-mel (B, T, n_mels)."""
        spec = torch.stft(
            wav, self.cfg.n_fft, hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length, window=self.window,
            center=True, return_complex=True,
        )
        power = spec.abs().pow(2)  # (B, n_freqs, T)
        mel = torch.einsum("mf,bft->btm", self.fb, power)
        return torch.log(mel.clamp_min(1e-5))
