"""Causal transformer building blocks shared by the encoder, decoder and heads.

- Pre-norm (RMSNorm) blocks with RoPE attention and SwiGLU FFNs.
- Strictly causal masking for the encoder / CTC head / probes.
- Block-wise causal masking for the streaming flow-matching decoder:
  full attention within a block, causal across blocks
  (StreamFlow / Qwen3-TTS-style chunked attention).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


def rope_frequencies(head_dim: int, seq_len: int, device: torch.device, base: float = 10_000.0) -> torch.Tensor:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64 (T, head_dim/2)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D). freqs: complex (T, D/2)."""
    b, h, t, d = x.shape
    xc = torch.view_as_complex(x.float().reshape(b, h, t, d // 2, 2))
    out = torch.view_as_real(xc * freqs.view(1, 1, t, d // 2))
    return out.reshape(b, h, t, d).to(x.dtype)


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """(T, T) bool mask, True = attend."""
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()


def block_causal_mask(seq_len: int, block_size: int, device: torch.device) -> torch.Tensor:
    """Full attention within a block, causal across blocks. True = attend."""
    blocks = torch.arange(seq_len, device=device) // block_size
    return blocks.unsqueeze(0) <= blocks.unsqueeze(1)  # key block <= query block


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask.view(1, 1, t, t),
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0):
        super().__init__()
        hidden = int(dim * mult * 2 / 3)
        hidden = (hidden + 7) // 8 * 8
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    """Pre-norm transformer block with optional adaLN-Zero conditioning.

    When `cond_dim` is set, a conditioning vector modulates the block via
    scale/shift/gate (DiT-style) - used by the flow-matching decoder for
    time-step / shortcut-size / speaker conditioning.
    """

    def __init__(self, dim: int, n_heads: int, ffn_mult: float = 4.0,
                 dropout: float = 0.0, cond_dim: Optional[int] = None):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_mult)
        self.adaln = None
        if cond_dim is not None:
            self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim))
            nn.init.zeros_(self.adaln[1].weight)
            nn.init.zeros_(self.adaln[1].bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, freqs: torch.Tensor,
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.adaln is not None and cond is not None:
            # cond: (B, D_c) global, or (B, T, D_c) per-frame.
            mod = self.adaln(cond)
            if mod.dim() == 2:
                mod = mod.unsqueeze(1)
            s1, b1, g1, s2, b2, g2 = mod.chunk(6, dim=-1)
            h = self.norm1(x) * (1 + s1) + b1
            x = x + g1 * self.attn(h, mask, freqs)
            h = self.norm2(x) * (1 + s2) + b2
            x = x + g2 * self.ffn(h)
        else:
            x = x + self.attn(self.norm1(x), mask, freqs)
            x = x + self.ffn(self.norm2(x))
        return x


class CausalTransformer(nn.Module):
    """A stack of causal blocks over (B, T, D) sequences."""

    def __init__(self, dim: int, n_layers: int, n_heads: int, ffn_mult: float = 4.0,
                 dropout: float = 0.0, cond_dim: Optional[int] = None,
                 block_size: Optional[int] = None):
        super().__init__()
        self.blocks = nn.ModuleList(
            Block(dim, n_heads, ffn_mult, dropout, cond_dim) for _ in range(n_layers)
        )
        self.norm = RMSNorm(dim)
        self.head_dim = dim // n_heads
        self.block_size = block_size  # None -> strictly causal

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        t = x.shape[1]
        freqs = rope_frequencies(self.head_dim, t, x.device)
        if self.block_size is None:
            mask = causal_mask(t, x.device)
        else:
            mask = block_causal_mask(t, self.block_size, x.device)
        for block in self.blocks:
            x = block(x, mask, freqs, cond)
        return self.norm(x)


class CausalConv1d(nn.Module):
    """Left-padded 1D convolution over (B, T, D): no lookahead."""

    def __init__(self, dim: int, kernel_size: int, out_dim: Optional[int] = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(dim, out_dim or dim, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))
        return self.conv(x).transpose(1, 2)
