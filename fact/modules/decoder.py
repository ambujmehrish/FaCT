"""Causal shortcut flow-matching mel decoder.

A DiT-style velocity-prediction network over mel frames with *block-wise
causal* attention (full attention within a block, causal across blocks), so
decoding streams block-by-block with bounded lookahead.

Stage A trains standard conditional flow matching (OT path). Stage B
fine-tunes with the shortcut self-consistency objective (Frans et al.;
SiTok's recipe: encoder + quantizer frozen, decoder distilled to 1-2 NFE).
The step size d is a first-class input, embedded alongside the flow time t:
d = 0 recovers plain flow matching; d = 1/2^k enables k-step sampling.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..config import AudioConfig, DecoderConfig
from .transformer import Block, RMSNorm, block_causal_mask, rope_frequencies


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """t: (B,) in [0, 1] -> (B, dim) sinusoidal embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device).float() / half
    )
    args = t.float().unsqueeze(-1) * freqs * 1000.0
    emb = torch.cat([args.cos(), args.sin()], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class FlowMatchingDecoder(nn.Module):
    def __init__(self, audio: AudioConfig, cfg: DecoderConfig,
                 cond_dim: int, spk_dim: int, frame_stack: int):
        super().__init__()
        self.cfg = cfg
        self.n_mels = audio.n_mels
        self.frame_stack = frame_stack

        self.x_proj = nn.Linear(audio.n_mels, cfg.dim)
        self.cond_proj = nn.Linear(cond_dim, cfg.dim)
        self.spk_proj = nn.Linear(spk_dim, cfg.dim)
        self.t_mlp = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim)
        )
        # Discrete step-size ids: 0 -> d=0 (plain FM), k -> d = 2^-(max_log2 - k + 1)
        # ... simplest scheme: id k in [0, max_shortcut_log2 + 1], id 0 reserved for d=0.
        self.d_embed = nn.Embedding(cfg.max_shortcut_log2 + 2, cfg.dim)
        nn.init.zeros_(self.d_embed.weight)

        self.blocks = nn.ModuleList(
            Block(cfg.dim, cfg.n_heads, cfg.ffn_mult, cfg.dropout, cond_dim=cfg.dim)
            for _ in range(cfg.n_layers)
        )
        self.norm = RMSNorm(cfg.dim)
        self.out = nn.Linear(cfg.dim, audio.n_mels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.head_dim = cfg.dim // cfg.n_heads

    def d_to_id(self, d: torch.Tensor) -> torch.Tensor:
        """Map step sizes to embedding ids. d=0 -> 0; d=2^-k -> max_log2 - k + 1."""
        ids = torch.zeros_like(d, dtype=torch.long)
        nz = d > 0
        k = torch.round(-torch.log2(d.clamp_min(1e-8))).long()
        ids[nz] = (self.cfg.max_shortcut_log2 - k[nz] + 1).clamp(1, self.cfg.max_shortcut_log2 + 1)
        return ids

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor,
                spk: torch.Tensor, d: Optional[torch.Tensor] = None,
                cond_drop_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict velocity.

        x_t:  (B, T_mel, n_mels) noisy mel.
        t:    (B,) flow time in [0, 1].
        cond: (B, T_tok, cond_dim) per-token condition at 12.5 Hz.
        spk:  (B, spk_dim) global speaker vector.
        d:    (B,) shortcut step size (0 = plain flow matching).
        cond_drop_mask: (B,) bool; True -> drop condition (CFG training).
        """
        b, t_mel, _ = x_t.shape
        c = self.cond_proj(cond)
        c = c.repeat_interleave(self.frame_stack, dim=1)[:, :t_mel]
        if c.shape[1] < t_mel:
            c = F.pad(c, (0, 0, 0, t_mel - c.shape[1]))
        if cond_drop_mask is not None:
            c = c * (~cond_drop_mask).float().view(b, 1, 1)

        if d is None:
            d = torch.zeros(b, device=x_t.device)
        ada = self.t_mlp(timestep_embedding(t, self.cfg.dim)) \
            + self.spk_proj(spk) + self.d_embed(self.d_to_id(d))

        h = self.x_proj(x_t) + c
        if self.cfg.causal:
            mask = block_causal_mask(t_mel, self.cfg.block_size, x_t.device)
        else:
            mask = torch.ones(t_mel, t_mel, dtype=torch.bool, device=x_t.device)
        freqs = rope_frequencies(self.head_dim, t_mel, x_t.device)
        for block in self.blocks:
            h = block(h, mask, freqs, ada)
        return self.out(self.norm(h))

    # ------------------------------------------------------------------ losses

    @staticmethod
    def _masked_mse(err: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """err: (B, T, C); mask: (B, T) bool (True = valid frame) or None."""
        if mask is None:
            return err.mean()
        m = mask.float().unsqueeze(-1)
        return (err * m).sum() / (m.sum() * err.shape[-1]).clamp_min(1)

    def flow_matching_loss(self, mel: torch.Tensor, cond: torch.Tensor,
                           spk: torch.Tensor,
                           mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Stage A: conditional flow matching on the OT path x_t = (1-t) e + t x1.

        mask: (B, T_mel) bool, True = real frame (padding excluded from loss).
        """
        b = mel.shape[0]
        t = torch.rand(b, device=mel.device)
        noise = torch.randn_like(mel)
        x_t = (1 - t.view(b, 1, 1)) * noise + t.view(b, 1, 1) * mel
        target_v = mel - noise
        drop = torch.rand(b, device=mel.device) < self.cfg.cfg_dropout
        v = self(x_t, t, cond, spk, cond_drop_mask=drop)
        return self._masked_mse((v - target_v).pow(2), mask)

    def shortcut_loss(self, mel: torch.Tensor, cond: torch.Tensor,
                      spk: torch.Tensor, fm_fraction: float = 0.75,
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Stage B: mix of plain FM (d=0) and self-consistency across step sizes.

        Self-consistency: one step of size 2d must match two chained steps of
        size d (targets computed with stop-gradient).
        """
        b = mel.shape[0]
        n_fm = max(1, int(b * fm_fraction))
        loss_fm = self.flow_matching_loss(
            mel[:n_fm], cond[:n_fm], spk[:n_fm],
            mask[:n_fm] if mask is not None else None,
        )
        if n_fm >= b:
            return loss_fm

        mel_c, cond_c, spk_c = mel[n_fm:], cond[n_fm:], spk[n_fm:]
        mask_c = mask[n_fm:] if mask is not None else None
        bc = mel_c.shape[0]
        # d = 2^-k with k in [1, max_log2]; take a step at time t = m * 2d.
        k = torch.randint(1, self.cfg.max_shortcut_log2 + 1, (bc,), device=mel.device)
        d = 2.0 ** (-k.float())
        max_m = (1.0 / (2 * d)).long().clamp_min(1)
        m = (torch.rand(bc, device=mel.device) * max_m.float()).floor().long()
        t = (m.float() * 2 * d).clamp(0.0, 1.0 - 2e-3)

        noise = torch.randn_like(mel_c)
        x_t = (1 - t.view(bc, 1, 1)) * noise + t.view(bc, 1, 1) * mel_c
        with torch.no_grad():
            v1 = self(x_t, t, cond_c, spk_c, d=d)
            x_mid = x_t + d.view(bc, 1, 1) * v1
            v2 = self(x_mid, t + d, cond_c, spk_c, d=d)
            target = 0.5 * (v1 + v2)
        v = self(x_t, t, cond_c, spk_c, d=2 * d)
        return loss_fm + self._masked_mse((v - target).pow(2), mask_c)

    # ---------------------------------------------------------------- sampling

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, spk: torch.Tensor, nfe: int = 2,
               cfg_scale: float = 1.0, temperature: float = 1.0) -> torch.Tensor:
        """Euler sampling with `nfe` steps using the matching shortcut size."""
        b, t_tok, _ = cond.shape
        t_mel = t_tok * self.frame_stack
        x = torch.randn(b, t_mel, self.n_mels, device=cond.device) * temperature
        d_val = 1.0 / nfe
        d = torch.full((b,), d_val, device=cond.device)
        for i in range(nfe):
            t = torch.full((b,), i * d_val, device=cond.device)
            v = self(x, t, cond, spk, d=d)
            if cfg_scale != 1.0:
                v_uncond = self(x, t, cond, spk, d=d,
                                cond_drop_mask=torch.ones(b, dtype=torch.bool, device=cond.device))
                v = v_uncond + cfg_scale * (v - v_uncond)
            x = x + d_val * v
        return x
