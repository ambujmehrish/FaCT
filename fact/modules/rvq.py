"""Residual vector quantization (EMA codebooks) - the quantizer-swap ablation.

Replaces the content FSQ at matched bits (default 91 x 91 = 8281 codes ~
2^13.02 vs FSQ's 2^13) to test 2509.20060's FSQ-over-RVQ finding at the
tokenizer level. Per-frame indices are FLATTENED across codebooks into one
integer (mixed-radix), so every downstream interface (LM probe,
embed_indices, detokenize) is unchanged.

EMA specifics:
- Laplace-smoothed cluster counts (van den Oord-style VQ-EMA).
- Dead codes (EMA count < threshold) are reseeded from batch vectors.
- Under DDP, count/sum statistics are all-reduced and reseeds broadcast
  from rank 0, so codebooks stay identical across ranks.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class RVQOutput(NamedTuple):
    continuous: torch.Tensor  # pre-quantization z (unbounded, unlike FSQ)
    quantized: torch.Tensor   # sum of codebook vectors, STE gradients
    indices: torch.Tensor     # flattened mixed-radix index, [0, prod(sizes))
    loss: torch.Tensor        # commitment loss (already weighted)


def _ddp_active() -> bool:
    return dist.is_available() and dist.is_initialized()


class ResidualVQ(nn.Module):
    def __init__(self, dim: int, codebook_sizes: Sequence[int],
                 decay: float = 0.99, commitment: float = 0.25,
                 eps: float = 1e-5, reseed_threshold: float = 1.0):
        super().__init__()
        self.dim = dim
        self.sizes = list(codebook_sizes)
        self.decay = decay
        self.commitment = commitment
        self.eps = eps
        self.reseed_threshold = reseed_threshold
        sizes_t = torch.tensor(self.sizes, dtype=torch.long)
        basis = torch.cumprod(
            torch.cat([torch.ones(1, dtype=torch.long), sizes_t[:-1]]), dim=0
        )
        self.register_buffer("basis", basis, persistent=False)
        for k, size in enumerate(self.sizes):
            embed = torch.randn(size, dim) * 0.1
            self.register_buffer(f"embed_{k}", embed)
            self.register_buffer(f"cluster_size_{k}", torch.ones(size))
            self.register_buffer(f"embed_avg_{k}", embed.clone())

    @property
    def codebook_size(self) -> int:
        out = 1
        for s in self.sizes:
            out *= s
        return out

    def _embed(self, k: int) -> torch.Tensor:
        return getattr(self, f"embed_{k}")

    @torch.no_grad()
    def _ema_update(self, k: int, residual: torch.Tensor, idx: torch.Tensor) -> None:
        size = self.sizes[k]
        onehot = F.one_hot(idx, size).to(residual.dtype)
        counts = onehot.sum(dim=0)
        sums = onehot.t() @ residual
        if _ddp_active():
            dist.all_reduce(counts)
            dist.all_reduce(sums)
        cluster = getattr(self, f"cluster_size_{k}")
        avg = getattr(self, f"embed_avg_{k}")
        cluster.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        avg.mul_(self.decay).add_(sums, alpha=1 - self.decay)
        n = cluster.sum()
        smoothed = (cluster + self.eps) / (n + size * self.eps) * n
        embed = getattr(self, f"embed_{k}")
        embed.copy_(avg / smoothed.unsqueeze(1))
        # Reseed dead codes from batch vectors (rank-0 choice under DDP).
        dead = cluster < self.reseed_threshold
        if bool(dead.any()) and residual.shape[0] > 0:
            n_dead = int(dead.sum())
            pick = torch.randint(0, residual.shape[0], (n_dead,),
                                 device=residual.device)
            samples = residual[pick]
            if _ddp_active():
                dist.broadcast(samples, src=0)
            embed[dead] = samples
            cluster[dead] = self.reseed_threshold
            avg[dead] = samples * self.reseed_threshold

    def forward(self, z: torch.Tensor) -> RVQOutput:
        """z: (..., dim) -> RVQOutput with flattened indices (...)."""
        shape = z.shape[:-1]
        flat = z.reshape(-1, self.dim)
        residual = flat
        q_total = torch.zeros_like(flat)
        flat_index = torch.zeros(flat.shape[0], dtype=torch.long, device=z.device)
        for k in range(len(self.sizes)):
            embed = self._embed(k)
            dist_sq = (residual.pow(2).sum(1, keepdim=True)
                       - 2 * residual @ embed.t()
                       + embed.pow(2).sum(1))
            idx = dist_sq.argmin(dim=1)
            if self.training:
                self._ema_update(k, residual.detach(), idx)
                embed = self._embed(k)  # post-update codebook
            q = embed[idx]
            flat_index = flat_index + idx * self.basis[k]
            q_total = q_total + q
            residual = residual - q.detach()
        quantized = flat + (q_total - flat).detach()
        commit = self.commitment * F.mse_loss(flat, q_total.detach())
        return RVQOutput(
            continuous=z,
            quantized=quantized.reshape(*shape, self.dim),
            indices=flat_index.reshape(shape),
            loss=commit,
        )

    @torch.no_grad()
    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Flattened indices -> summed codebook vectors (..., dim)."""
        sizes = torch.tensor(self.sizes, dtype=torch.long, device=indices.device)
        out = torch.zeros(*indices.shape, self.dim,
                          device=indices.device, dtype=self._embed(0).dtype)
        for k in range(len(self.sizes)):
            idx_k = (indices // self.basis[k]) % sizes[k]
            out = out + self._embed(k)[idx_k]
        return out
