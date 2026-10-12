"""X-codec2 (LLaSA line) - the single-codebook semantic-fusion school:
50 Hz, one 65k codebook, semantic+acoustic fused. The strongest "one merged
stream" counterpoint to factorization.

Requires `pip install xcodec2` (heavy dependency set - install in a
separate venv if it conflicts) and its HF checkpoint pre-downloaded.
Metadata is read from the loaded model where exposed.
"""

from __future__ import annotations

import math

import torch

from .base import register


class XCodec2Baseline:
    name = "xcodec2"

    def __init__(self, model_id: str = "HKUSTAudio/xcodec2", device: str = "cuda"):
        try:
            from xcodec2.modeling_xcodec2 import XCodec2Model
        except ImportError as e:
            raise ImportError("xcodec2 baseline requires `pip install xcodec2`") from e
        self.model = XCodec2Model.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.sample_rate = 16_000
        self.frame_rate_hz = 50.0
        codebook_size = 65_536
        self.vocab_sizes = [codebook_size]
        self.discrete_bps = math.log2(codebook_size) * self.frame_rate_hz

    @torch.no_grad()
    def tokenize(self, wav: torch.Tensor) -> list[torch.Tensor]:
        codes = self.model.encode_code(input_waveform=wav.to(self.device))
        return [codes.reshape(wav.shape[0], -1).long()]

    @torch.no_grad()
    def reconstruct(self, wav: torch.Tensor) -> torch.Tensor:
        codes = self.model.encode_code(input_waveform=wav.to(self.device))
        return self.model.decode_code(codes).squeeze(1)


@register("xcodec2")
def _build(device: str = "cuda", model_id: str = "HKUSTAudio/xcodec2") -> XCodec2Baseline:
    return XCodec2Baseline(model_id=model_id, device=device)
