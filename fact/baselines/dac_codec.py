"""Descript Audio Codec (DAC) - the pure-fidelity ceiling reference:
high-bitrate RVQ, no semantic constraint, non-causal. Anchors the top of
the reconstruction axis so low-bitrate comparisons stay honest.

Requires `pip install descript-audio-codec` and a pre-downloaded weights
file (login node: `python scripts/download_checkpoints.py --baselines dac`).

Metadata (hop, codebooks, codebook size, bitrate) is read from the loaded
model - nothing hardcoded.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .base import register


class DACBaseline:
    name = "dac"

    def __init__(self, model_type: str = "24khz", device: str = "cuda",
                 weights_path: Optional[str] = None):
        try:
            import dac
        except ImportError as e:
            raise ImportError("dac baseline requires `pip install descript-audio-codec`") from e
        if weights_path is None:
            weights_path = dac.utils.download(model_type=model_type)  # cached; login node
        self.model = dac.DAC.load(str(weights_path)).to(device).eval()
        self.device = device
        self.sample_rate = int(self.model.sample_rate)
        self.frame_rate_hz = self.sample_rate / self.model.hop_length
        n_books = int(self.model.n_codebooks)
        codebook_size = int(self.model.codebook_size)
        self.vocab_sizes = [codebook_size] * n_books
        self.discrete_bps = n_books * math.log2(codebook_size) * self.frame_rate_hz

    @torch.no_grad()
    def _encode(self, wav: torch.Tensor):
        x = self.model.preprocess(wav.unsqueeze(1).to(self.device), self.sample_rate)
        _, codes, latents, _, _ = self.model.encode(x)
        return codes  # (B, n_codebooks, T)

    @torch.no_grad()
    def tokenize(self, wav: torch.Tensor) -> list[torch.Tensor]:
        codes = self._encode(wav)
        return [codes[:, k].long() for k in range(codes.shape[1])]

    @torch.no_grad()
    def reconstruct(self, wav: torch.Tensor) -> torch.Tensor:
        codes = self._encode(wav)
        z = self.model.quantizer.from_codes(codes)[0]
        return self.model.decode(z).squeeze(1)


@register("dac")
def _build(device: str = "cuda", model_type: str = "24khz",
           weights_path: Optional[str] = None) -> DACBaseline:
    return DACBaseline(model_type=model_type, device=device, weights_path=weights_path)
