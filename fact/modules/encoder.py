"""Causal encoder: mel @ 50 Hz -> latent frames @ 12.5 Hz.

A causal conv frontend and transformer layers run at the mel rate, then
`frame_stack` consecutive frames are stacked channel-wise (50 -> 12.5 Hz)
and a second causal transformer stack refines the low-rate latents.
Everything is strictly causal: latent i depends only on mel frames
< (i + 1) * frame_stack.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import AudioConfig, EncoderConfig
from .transformer import CausalConv1d, CausalTransformer


class CausalEncoder(nn.Module):
    def __init__(self, audio: AudioConfig, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(audio.n_mels, cfg.dim)
        self.frontend_conv = CausalConv1d(cfg.dim, cfg.conv_kernel)
        self.pre = CausalTransformer(
            cfg.dim, cfg.n_layers_pre, cfg.n_heads, cfg.ffn_mult, cfg.dropout
        )
        self.stack_proj = nn.Linear(cfg.dim * cfg.frame_stack, cfg.dim)
        self.post = CausalTransformer(
            cfg.dim, cfg.n_layers_post, cfg.n_heads, cfg.ffn_mult, cfg.dropout
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, T_mel, n_mels) -> (B, T_mel // frame_stack, dim)."""
        b, t, _ = mel.shape
        s = self.cfg.frame_stack
        t_trim = (t // s) * s
        x = self.in_proj(mel[:, :t_trim])
        x = x + self.frontend_conv(x)
        x = self.pre(x)
        x = x.reshape(b, t_trim // s, s * self.cfg.dim)
        x = self.stack_proj(x)
        return self.post(x)
