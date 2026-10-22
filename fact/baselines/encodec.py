"""EnCodec 24 kHz (Defossez et al., 2022): the classic causal streaming codec.

    encodec:bandwidth=1.5   (also 3.0, 6.0, 12.0, 24.0 kbps)

The 24 kHz model is causal (use_causal_conv in its config). Loaded through
transformers from the HF cache (facebook/encodec_24khz). Frame rate is
measured with a probe and the number of codebooks read from the codes, so
the reported bitrate is the checkpoint's own.
"""

from __future__ import annotations

import math

import torch

from .base import register

MODEL_ID = "facebook/encodec_24khz"


class EncodecBaseline:
    def __init__(self, bandwidth: float = 1.5, device: str = "cuda"):
        from transformers import EncodecModel
        from .lowrate import measured_frame_rate
        self.model = EncodecModel.from_pretrained(MODEL_ID).to(device).eval()
        self.bw = float(bandwidth)
        if self.bw not in self.model.config.target_bandwidths:
            raise ValueError(f"bandwidth must be one of {self.model.config.target_bandwidths}")
        self.name = f"encodec:{self.bw}"
        self.device = device
        self.sample_rate = self.model.config.sampling_rate
        self.frame_rate_hz = measured_frame_rate(lambda x: self._codes(x), self.sample_rate, device)
        probe = torch.zeros(1, self.sample_rate, device=device)
        with torch.no_grad():
            n_q = self._codes(probe).shape[-2]
        cb = int(self.model.config.codebook_size)
        self.vocab_sizes = [cb] * n_q
        self.discrete_bps = n_q * math.log2(cb) * self.frame_rate_hz

    def _codes(self, wav):
        return self.model.encode(wav.unsqueeze(1), bandwidth=self.bw).audio_codes  # (1, B, n_q, T)

    @torch.no_grad()
    def tokenize(self, wav):
        codes = self._codes(wav.to(self.device))[0]
        return [codes[:, q] for q in range(codes.shape[1])]

    @torch.no_grad()
    def reconstruct(self, wav):
        wav = wav.to(self.device)
        enc = self.model.encode(wav.unsqueeze(1), bandwidth=self.bw)
        out = self.model.decode(enc.audio_codes, enc.audio_scales)[0]
        return out[:, 0, : wav.shape[-1]]


@register("encodec")
def _build(device: str = "cuda", bandwidth: float = 1.5) -> EncodecBaseline:
    return EncodecBaseline(bandwidth=bandwidth, device=device)
