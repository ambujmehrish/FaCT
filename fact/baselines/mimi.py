"""Mimi (Moshi's tokenizer, kyutai/mimi) - the closest external baseline:
12.5 Hz, causal encoder+decoder, RVQ with a WavLM-distilled semantic first
codebook. Everything FaCT does, minus the semi-discrete duality and the
prosody factorization.

Requires `pip install transformers` and a pre-downloaded checkpoint
(login node: `python scripts/download_checkpoints.py --baselines mimi`).

All metadata (frame rate, codebook size, bitrate) is read from the model
config - nothing hardcoded, so reported numbers are verifiable.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .base import register


class MimiTokenStreams:
    name = "mimi"

    def __init__(self, model_id: str = "kyutai/mimi", n_streams: int = 8,
                 device: str = "cuda"):
        try:
            from transformers import MimiModel
        except ImportError as e:
            raise ImportError("mimi baseline requires `pip install transformers`") from e
        self.model = MimiModel.from_pretrained(model_id).to(device).eval()
        self.n_streams = n_streams
        self.device = device
        cfg = self.model.config
        self.sample_rate = int(cfg.sampling_rate)
        self.frame_rate_hz = float(cfg.frame_rate)
        codebook_size = int(cfg.codebook_size)
        self.vocab_sizes = [codebook_size] * n_streams
        self.discrete_bps = n_streams * math.log2(codebook_size) * self.frame_rate_hz

    @torch.no_grad()
    def _encode(self, wav: torch.Tensor) -> torch.Tensor:
        return self.model.encode(
            wav.unsqueeze(1).to(self.device), num_quantizers=self.n_streams
        ).audio_codes  # (B, n_streams, T)

    @torch.no_grad()
    def tokenize(self, wav: torch.Tensor) -> list[torch.Tensor]:
        """wav: (B, N) mono at self.sample_rate -> n_streams (B, T) tensors."""
        codes = self._encode(wav)
        return [codes[:, k].long() for k in range(self.n_streams)]

    @torch.no_grad()
    def reconstruct(self, wav: torch.Tensor) -> torch.Tensor:
        codes = self._encode(wav)
        audio = self.model.decode(codes).audio_values  # (B, 1, N)
        return audio.squeeze(1)


@register("mimi")
def _build(device: str = "cuda", n_streams: int = 8,
           model_id: str = "kyutai/mimi") -> MimiTokenStreams:
    return MimiTokenStreams(model_id=model_id, n_streams=n_streams, device=device)
