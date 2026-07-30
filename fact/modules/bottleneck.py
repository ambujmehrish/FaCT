"""Factorized semi-discrete bottleneck.

Three parallel channels over each 12.5 Hz encoder frame:

- CONTENT head: large FSQ lattice; semantically grounded by the CTC head.
- PROSODY head: small FSQ lattice; supervised by F0/energy targets.
- RESIDUAL channel (optional): low-dim continuous, KL-regularized toward
  N(0, I) (HoliTok-style variational smoothness) so it stays learnable by
  a downstream flow head. This is the honest fidelity-ceiling escape hatch.

Every channel exposes both views of the same state: a continuous embedding
(for the decoder / continuous downstream interface) and, for the FSQ heads,
integer indices (for the discrete downstream interface).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ..config import BottleneckConfig
from .fsq import FSQ, FSQOutput


@dataclass
class BottleneckOutput:
    # Per-frame embeddings projected back to model dim (decoder consumables).
    content_emb: torch.Tensor        # (B, T, dim) from quantized (STE) content codes
    prosody_emb: torch.Tensor        # (B, T, dim) from quantized (STE) prosody codes
    residual_emb: Optional[torch.Tensor]  # (B, T, dim) or None
    # Discrete view.
    content_indices: torch.Tensor    # (B, T) long
    prosody_indices: torch.Tensor    # (B, T) long
    # Continuous (pre-round) view - the semi-discrete duality.
    content_continuous: torch.Tensor  # (B, T, |content_levels|) in [-1, 1]
    prosody_continuous: torch.Tensor  # (B, T, |prosody_levels|) in [-1, 1]
    residual: Optional[torch.Tensor]  # (B, T, residual_dim) sampled latent
    # Raw quantized lattice points (normalized), for probes.
    content_quantized: torch.Tensor
    prosody_quantized: torch.Tensor
    kl_loss: torch.Tensor            # scalar; zero if residual disabled

    def decoder_condition(self) -> torch.Tensor:
        """Sum of channel embeddings: the decoder's per-frame condition."""
        cond = self.content_emb + self.prosody_emb
        if self.residual_emb is not None:
            cond = cond + self.residual_emb
        return cond


class FactorizedBottleneck(nn.Module):
    def __init__(self, dim: int, cfg: BottleneckConfig):
        super().__init__()
        self.cfg = cfg
        self.content_fsq = FSQ(cfg.content_levels)
        self.prosody_fsq = FSQ(cfg.prosody_levels)

        self.content_down = nn.Linear(dim, self.content_fsq.num_dims)
        self.content_up = nn.Linear(self.content_fsq.num_dims, dim)
        self.prosody_down = nn.Linear(dim, self.prosody_fsq.num_dims)
        self.prosody_up = nn.Linear(self.prosody_fsq.num_dims, dim)

        self.residual_dim = cfg.residual_dim
        if cfg.residual_dim > 0:
            self.residual_down = nn.Linear(dim, 2 * cfg.residual_dim)  # mu, logvar
            self.residual_up = nn.Linear(cfg.residual_dim, dim)
        else:
            self.residual_down = None
            self.residual_up = None

    @property
    def content_codebook_size(self) -> int:
        return self.content_fsq.codebook_size

    @property
    def prosody_codebook_size(self) -> int:
        return self.prosody_fsq.codebook_size

    def forward(self, h: torch.Tensor) -> BottleneckOutput:
        """h: (B, T, dim) encoder output at 12.5 Hz."""
        c: FSQOutput = self.content_fsq(self.content_down(h))
        p: FSQOutput = self.prosody_fsq(self.prosody_down(h))

        content_emb = self.content_up(c.quantized)
        prosody_emb = self.prosody_up(p.quantized)

        residual = None
        residual_emb = None
        kl = h.new_zeros(())
        if self.residual_down is not None:
            mu, logvar = self.residual_down(h).chunk(2, dim=-1)
            logvar = logvar.clamp(-8.0, 8.0)
            if self.training:
                residual = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            else:
                residual = mu
            kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).mean()
            residual_emb = self.residual_up(residual)

        return BottleneckOutput(
            content_emb=content_emb,
            prosody_emb=prosody_emb,
            residual_emb=residual_emb,
            content_indices=c.indices,
            prosody_indices=p.indices,
            content_continuous=c.continuous,
            prosody_continuous=p.continuous,
            residual=residual,
            content_quantized=c.quantized,
            prosody_quantized=p.quantized,
            kl_loss=kl,
        )

    def embed_indices(self, content_indices: torch.Tensor,
                      prosody_indices: torch.Tensor,
                      residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Rebuild the decoder condition purely from the discrete view.

        This is the detokenization path an index-predicting LM uses; the
        optional residual comes from a downstream continuous head (or is
        omitted, accepting the fidelity ceiling of the pure-discrete view).
        """
        c = self.content_fsq.indices_to_codes(content_indices)
        p = self.prosody_fsq.indices_to_codes(prosody_indices)
        cond = self.content_up(c) + self.prosody_up(p)
        if residual is not None and self.residual_up is not None:
            cond = cond + self.residual_up(residual)
        return cond
