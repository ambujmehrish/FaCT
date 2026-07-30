"""Mimi (Moshi's tokenizer) as an external baseline for the modelability probe.

Wraps the pretrained `kyutai/mimi` checkpoint from HuggingFace transformers
and exposes the same interface the probe consumes: per-frame integer index
streams (one per RVQ codebook) + a frame rate. No retraining - Mimi is a
public-checkpoint reference point (as are X-codec2 / WavTokenizer /
TaDiCodec; wrap them the same way).

Requires: pip install transformers  (and network/HF cache for the weights).

Usage:
    from fact.baselines.mimi import MimiTokenStreams
    mimi = MimiTokenStreams(n_streams=8)
    streams = mimi.tokenize(wav_24k)          # list of (B, T) long tensors
    # -> feed to fact.eval.modelability.TokenLMProbe(mimi.vocab_sizes, ...)
"""

from __future__ import annotations

import torch


class MimiTokenStreams:
    frame_rate_hz: float = 12.5
    sample_rate: int = 24_000

    def __init__(self, model_id: str = "kyutai/mimi", n_streams: int = 8,
                 device: str = "cpu"):
        try:
            from transformers import MimiModel
        except ImportError as e:
            raise ImportError(
                "MimiTokenStreams requires `pip install transformers`"
            ) from e
        self.model = MimiModel.from_pretrained(model_id).to(device).eval()
        self.n_streams = n_streams
        self.device = device
        codebook_size = self.model.config.codebook_size
        self.vocab_sizes = [codebook_size] * n_streams

    @torch.no_grad()
    def tokenize(self, wav: torch.Tensor) -> list[torch.Tensor]:
        """wav: (B, N) mono at 24 kHz -> list of n_streams (B, T) index tensors."""
        codes = self.model.encode(
            wav.unsqueeze(1).to(self.device),
            num_quantizers=self.n_streams,
        ).audio_codes  # (B, n_streams, T)
        return [codes[:, k].long().cpu() for k in range(self.n_streams)]
