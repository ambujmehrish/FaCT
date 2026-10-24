"""Causal encoder: mel @ 50 Hz -> latent frames @ 12.5 Hz.

A causal conv frontend and transformer layers run at the mel rate, then
`frame_stack` consecutive frames are stacked channel-wise (50 -> 12.5 Hz)
and a second causal transformer stack refines the low-rate latents.
Everything is strictly causal: latent i depends only on mel frames
< (i + 1) * frame_stack. With encoder.chunk_tokens = c > 1 attention is
full inside aligned chunks of c tokens: latent i depends on mel frames
< (i // c + 1) * c * frame_stack (bounded lookahead, still streamable).
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
        self.ssl_proj = nn.Linear(cfg.ssl_dim, cfg.dim) if cfg.ssl_model else None
        c = cfg.chunk_tokens
        if c < 1:
            raise ValueError(f"encoder.chunk_tokens must be >= 1, got {c}")
        if c > 1 and not cfg.causal:
            raise ValueError("encoder.chunk_tokens > 1 needs encoder.causal=True")
        # chunk_tokens=1 keeps the strictly causal masks (block_size None).
        pre_block = c * cfg.frame_stack if c > 1 else None
        post_block = c if c > 1 else None
        self.frontend_conv = CausalConv1d(cfg.dim, cfg.conv_kernel, causal=cfg.causal)
        self.pre = CausalTransformer(
            cfg.dim, cfg.n_layers_pre, cfg.n_heads, cfg.ffn_mult, cfg.dropout,
            block_size=pre_block, causal=cfg.causal,
        )
        self.stack_proj = nn.Linear(cfg.dim * cfg.frame_stack, cfg.dim)
        self.post = CausalTransformer(
            cfg.dim, cfg.n_layers_post, cfg.n_heads, cfg.ffn_mult, cfg.dropout,
            block_size=post_block, causal=cfg.causal,
        )

    def forward(self, mel: torch.Tensor, ssl: torch.Tensor | None = None) -> torch.Tensor:
        """mel: (B, T_mel, n_mels) [+ ssl: (B, T_mel, ssl_dim)] -> (B, T_mel // frame_stack, dim)."""
        b, t, _ = mel.shape
        s = self.cfg.frame_stack
        t_trim = (t // s) * s
        x = self.in_proj(mel[:, :t_trim])
        if self.ssl_proj is not None:
            if ssl is None:
                raise ValueError(f"encoder expects {self.cfg.ssl_model} features (ssl=None)")
            if ssl.shape[1] < t_trim:
                raise ValueError(f"ssl features have {ssl.shape[1]} frames, mel needs {t_trim}")
            x = x + self.ssl_proj(ssl[:, :t_trim].to(x.dtype))
        elif ssl is not None:
            raise ValueError("ssl features given to an encoder without ssl_model")
        x = x + self.frontend_conv(x)
        x = self.pre(x)
        x = x.reshape(b, t_trim // s, s * self.cfg.dim)
        x = self.stack_proj(x)
        return self.post(x)
