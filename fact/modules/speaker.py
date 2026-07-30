"""Global speaker reference encoder.

Speaker identity deliberately lives in neither token stream: it is provided
to the decoder as a single global conditioning vector computed from a
reference mel (the responsible-release choice - voice cloning requires
explicit reference conditioning, and factorization stays speaker-free).
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import AudioConfig, SpeakerConfig
from .transformer import CausalConv1d


class SpeakerEncoder(nn.Module):
    def __init__(self, audio: AudioConfig, cfg: SpeakerConfig):
        super().__init__()
        self.cfg = cfg
        dims = [audio.n_mels] + [cfg.dim] * cfg.n_layers
        self.convs = nn.ModuleList(
            CausalConv1d(dims[i], kernel_size=3, out_dim=dims[i + 1])
            for i in range(cfg.n_layers)
        )
        self.act = nn.SiLU()
        self.out = nn.Linear(cfg.dim, cfg.dim)

    def forward(self, ref_mel: torch.Tensor) -> torch.Tensor:
        """ref_mel: (B, T_ref, n_mels) -> (B, dim) L2-normalized."""
        x = ref_mel
        for conv in self.convs:
            x = self.act(conv(x))
        x = x.mean(dim=1)  # temporal average pool
        x = self.out(x)
        return torch.nn.functional.normalize(x, dim=-1)
