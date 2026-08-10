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

    def forward(self, ref_mel: torch.Tensor,
                lengths: torch.Tensor | None = None) -> torch.Tensor:
        """ref_mel: (B, T_ref, n_mels) -> (B, dim) L2-normalized.

        lengths: per-item valid frame counts; padded frames are excluded
        from the temporal pool so batch padding cannot dilute the speaker
        vector.
        """
        x = ref_mel
        for conv in self.convs:
            x = self.act(conv(x))
        if lengths is None:
            x = x.mean(dim=1)
        else:
            t = x.shape[1]
            m = (torch.arange(t, device=x.device).unsqueeze(0)
                 < lengths.clamp(max=t).unsqueeze(1)).float().unsqueeze(-1)
            x = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1)
        x = self.out(x)
        return torch.nn.functional.normalize(x, dim=-1)
