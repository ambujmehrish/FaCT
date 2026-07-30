"""Synthetic speech-like data for smoke tests and CI.

Generates harmonic signals with piecewise-constant "phoneme" segments (each
segment has a text label and a formant-like spectral signature) and a
wandering F0 contour. This exercises every loss in the model - CTC has real
(if trivial) structure to latch onto, and F0 targets correlate with the
signal - without any audio dependencies.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import Dataset

from ..config import FaCTConfig
from ..model import Batch
from ..modules.mel import LogMelSpectrogram
from .prosody_targets import prosody_targets_from_mel_and_f0


class SyntheticSpeech(Dataset):
    def __init__(self, cfg: FaCTConfig, n_items: int = 64,
                 seconds: float = 2.0, seed: int = 0):
        self.cfg = cfg
        self.n_items = n_items
        self.n_samples = int(seconds * cfg.audio.sample_rate)
        self.seed = seed
        self.mel = LogMelSpectrogram(cfg.audio)

    def __len__(self) -> int:
        return self.n_items

    def _make_wav(self, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sr = self.cfg.audio.sample_rate
        n = self.n_samples
        n_segs = int(torch.randint(3, 7, (1,), generator=g))
        bounds = torch.sort(torch.randint(1, n - 1, (n_segs - 1,), generator=g)).values
        bounds = torch.cat([torch.zeros(1, dtype=torch.long), bounds,
                            torch.tensor([n])])
        # Per-segment "phoneme" id in [1, vocab) (0 is CTC blank).
        vocab = self.cfg.ctc.vocab_size
        labels = torch.randint(1, vocab, (n_segs,), generator=g)

        f0_base = 100.0 + 150.0 * torch.rand(1, generator=g).item()
        t = torch.arange(n) / sr
        f0_track = f0_base * (1.0 + 0.2 * torch.sin(2 * torch.pi * 2.0 * t))

        wav = torch.zeros(n)
        phase = torch.cumsum(2 * torch.pi * f0_track / sr, dim=0)
        for i in range(n_segs):
            lo, hi = int(bounds[i]), int(bounds[i + 1])
            if hi <= lo:
                continue
            seg = torch.zeros(hi - lo)
            # Harmonic stack whose amplitudes depend on the label (formant proxy).
            for h in range(1, 5):
                amp = 0.5 / h * (1.0 + 0.5 * torch.cos(torch.tensor(float(labels[i] * h))))
                seg = seg + amp * torch.sin(h * phase[lo:hi])
            wav[lo:hi] = seg
        wav = wav / wav.abs().max().clamp_min(1e-5) * 0.7

        # F0 at the mel frame rate (0 where the wav is near-silent).
        hop = self.cfg.audio.hop_length
        n_frames = n // hop + 1
        idx = (torch.arange(n_frames) * hop).clamp_max(n - 1)
        f0_frames = f0_track[idx]
        return wav, f0_frames, labels

    def __getitem__(self, i: int) -> dict:
        g = torch.Generator().manual_seed(self.seed * 100_003 + i)
        wav, f0_frames, labels = self._make_wav(g)
        with torch.no_grad():
            mel = self.mel(wav.unsqueeze(0)).squeeze(0)
        t = min(mel.shape[0], f0_frames.shape[0])
        return {"mel": mel[:t], "f0": f0_frames[:t], "text": labels}


def collate(items: list[dict], cfg: FaCTConfig) -> Batch:
    t_min = min(it["mel"].shape[0] for it in items)
    s = cfg.encoder.frame_stack
    t_min = (t_min // s) * s
    mel = torch.stack([it["mel"][:t_min] for it in items])
    f0 = torch.stack([it["f0"][:t_min] for it in items])
    f0_bins, energy = prosody_targets_from_mel_and_f0(mel, f0, cfg.prosody, s)

    s_max = max(it["text"].shape[0] for it in items)
    text = torch.zeros(len(items), s_max, dtype=torch.long)
    text_lengths = torch.zeros(len(items), dtype=torch.long)
    for b, it in enumerate(items):
        text[b, : it["text"].shape[0]] = it["text"]
        text_lengths[b] = it["text"].shape[0]

    return Batch(mel=mel, text=text, text_lengths=text_lengths,
                 f0_bins=f0_bins, energy=energy)
