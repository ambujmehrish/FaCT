"""Finite Scalar Quantization with a first-class semi-discrete dual view.

FSQ (Mentzer et al., 2023) bounds each latent dimension to a small number of
levels and rounds to the nearest level with a straight-through estimator.
There is no codebook to collapse, and every quantized vector has a unique
integer index.

FaCT makes the duality an interface rather than an internal trick
(VoxCPM-style FSQ-as-regularization):

- ``continuous``: the *bounded, pre-rounding* vector. This is the
  near-lossless continuous view a flow head can consume.
- ``quantized``: the rounded lattice point, with straight-through gradients.
- ``indices``: the unique integer id of the lattice point - the discrete
  view an LM can predict, exactly addressing the same state.

All views are normalized to [-1, 1] per dimension.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import torch
from torch import nn


class FSQOutput(NamedTuple):
    continuous: torch.Tensor  # bounded pre-round vector, in [-1, 1]
    quantized: torch.Tensor   # rounded lattice point (STE gradients), in [-1, 1]
    indices: torch.Tensor     # long tensor of lattice ids, in [0, codebook_size)


def _round_ste(z: torch.Tensor) -> torch.Tensor:
    return z + (z.round() - z).detach()


class FSQ(nn.Module):
    def __init__(self, levels: Sequence[int], eps: float = 1e-3):
        super().__init__()
        if any(l < 2 for l in levels):
            raise ValueError(f"All FSQ levels must be >= 2, got {levels}")
        self.eps = eps
        levels_t = torch.tensor(list(levels), dtype=torch.long)
        self.register_buffer("levels", levels_t, persistent=False)
        # Mixed-radix basis for index <-> code conversion.
        basis = torch.cumprod(
            torch.cat([torch.ones(1, dtype=torch.long), levels_t[:-1]]), dim=0
        )
        self.register_buffer("basis", basis, persistent=False)
        # half_width maps rounded integer codes back to [-1, 1].
        self.register_buffer(
            "half_width", (levels_t // 2).to(torch.float32), persistent=False
        )

    @property
    def num_dims(self) -> int:
        return int(self.levels.numel())

    @property
    def codebook_size(self) -> int:
        return int(torch.prod(self.levels).item())

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Bound z so that rounding lands on exactly `levels` integers per dim."""
        levels = self.levels.to(z.dtype)
        half_l = (levels - 1) * (1 + self.eps) / 2
        # Even level counts have no level at 0; shift the tanh so rounding
        # produces the correct asymmetric grid.
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0).to(z.dtype)
        shift = torch.atanh(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def _bounded_to_codes(self, z_bounded: torch.Tensor) -> torch.Tensor:
        """Rounded integer codes shifted to be non-negative: dim d in [0, L_d)."""
        half = (self.levels // 2).to(z_bounded.dtype)
        return z_bounded.round() + half

    def forward(self, z: torch.Tensor) -> FSQOutput:
        """z: (..., num_dims) unbounded latents."""
        if z.shape[-1] != self.num_dims:
            raise ValueError(f"Expected last dim {self.num_dims}, got {z.shape[-1]}")
        z_bounded = self.bound(z)
        z_q = _round_ste(z_bounded)
        continuous = z_bounded / self.half_width
        quantized = z_q / self.half_width
        codes = self._bounded_to_codes(z_bounded.detach())
        indices = (codes.to(torch.long) * self.basis).sum(dim=-1)
        return FSQOutput(continuous=continuous, quantized=quantized, indices=indices)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Map lattice ids back to normalized [-1, 1] vectors (..., num_dims)."""
        codes = (indices.unsqueeze(-1) // self.basis) % self.levels
        half = (self.levels // 2).to(torch.float32)
        return (codes.to(torch.float32) - half) / self.half_width
