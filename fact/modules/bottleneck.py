"""Factorized semi-discrete bottleneck - and its matched internal baselines.

variant "factorized" (FaCT): three parallel channels per 12.5 Hz frame:
- CONTENT head: large FSQ lattice; semantically grounded by the CTC head.
- PROSODY head: small FSQ lattice; supervised by F0/energy targets.
- RESIDUAL channel (optional): low-dim continuous, KL-regularized toward
  N(0, I) (HoliTok-style variational smoothness).

variant "single_fsq": one FSQ head over the union of the content and prosody
lattices (exactly matched bits and dims, no factorization) - the ablation
that isolates RQ2. The single stream lives in the `content_*` fields.

variant "vae": a pure-continuous KL-regularized latent (matched total dims,
no discrete view) - the matched continuous baseline for RQ1/RQ3. The latent
lives in the `residual`/`residual_emb` fields; all index fields are None.

Every discrete channel exposes both views of the same state: a continuous
embedding (decoder / continuous downstream interface) and integer indices
(discrete downstream interface).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ..config import BottleneckConfig
from .fsq import FSQ, FSQOutput
from .rvq import ResidualVQ, RVQOutput

VARIANTS = ("factorized", "single_fsq", "vae", "rvq")


@dataclass
class BottleneckOutput:
    # Per-frame embeddings projected back to model dim (decoder consumables).
    content_emb: Optional[torch.Tensor]   # (B, T, dim); None for "vae"
    prosody_emb: Optional[torch.Tensor]   # (B, T, dim); "factorized" only
    residual_emb: Optional[torch.Tensor]  # (B, T, dim) or None
    # Discrete view.
    content_indices: Optional[torch.Tensor]  # (B, T) long; None for "vae"
    prosody_indices: Optional[torch.Tensor]  # (B, T) long; "factorized" only
    # Continuous (pre-round) view - the semi-discrete duality.
    content_continuous: Optional[torch.Tensor]  # (B, T, n_dims) in [-1, 1]
    prosody_continuous: Optional[torch.Tensor]
    residual: Optional[torch.Tensor]      # (B, T, residual_dim/vae_dim) latent
    # Raw quantized lattice points (normalized), for probes.
    content_quantized: Optional[torch.Tensor]
    prosody_quantized: Optional[torch.Tensor]
    kl_loss: torch.Tensor  # per-frame (B, T); scalar 0 if no continuous channel
    quantizer_loss: torch.Tensor  # scalar; RVQ commitment (0 for FSQ variants)

    def decoder_condition(self) -> torch.Tensor:
        """Sum of channel embeddings: the decoder's per-frame condition."""
        parts = [e for e in (self.content_emb, self.prosody_emb, self.residual_emb)
                 if e is not None]
        cond = parts[0]
        for p in parts[1:]:
            cond = cond + p
        return cond

    def primary_emb(self) -> torch.Tensor:
        """The embedding CTC grounds: content stream, or the VAE latent."""
        return self.content_emb if self.content_emb is not None else self.residual_emb

    def index_streams(self) -> list[tuple[str, torch.Tensor]]:
        """Named discrete streams, for the modelability probe."""
        streams = []
        if self.content_indices is not None:
            streams.append(("content", self.content_indices))
        if self.prosody_indices is not None:
            streams.append(("prosody", self.prosody_indices))
        return streams


class FactorizedBottleneck(nn.Module):
    def __init__(self, dim: int, cfg: BottleneckConfig):
        super().__init__()
        if cfg.variant not in VARIANTS:
            raise ValueError(f"Unknown bottleneck variant {cfg.variant!r}; pick from {VARIANTS}")
        self.cfg = cfg
        self.variant = cfg.variant

        self.content_fsq = None
        self.content_rvq = None
        self.prosody_fsq = None
        if self.variant in ("factorized", "rvq"):
            # Both keep the full factorization; "rvq" only swaps the content
            # quantizer (the quantizer-type ablation at matched bits).
            if self.variant == "factorized":
                self.content_fsq = FSQ(cfg.content_levels)
                c_dim = self.content_fsq.num_dims
            else:
                self.content_rvq = ResidualVQ(
                    cfg.rvq_dim, cfg.rvq_codebook_sizes,
                    decay=cfg.rvq_decay, commitment=cfg.rvq_commitment,
                )
                c_dim = cfg.rvq_dim
            self.prosody_fsq = FSQ(cfg.prosody_levels)
            self.prosody_down = nn.Linear(dim, self.prosody_fsq.num_dims)
            self.prosody_up = nn.Linear(self.prosody_fsq.num_dims, dim)
            self.content_down = nn.Linear(dim, c_dim)
            self.content_up = nn.Linear(c_dim, dim)
        elif self.variant == "single_fsq":
            self.content_fsq = FSQ(cfg.single_levels)
            self.content_down = nn.Linear(dim, self.content_fsq.num_dims)
            self.content_up = nn.Linear(self.content_fsq.num_dims, dim)

        cont_dim = cfg.vae_dim if self.variant == "vae" else cfg.residual_dim
        self.residual_dim = cont_dim
        if cont_dim > 0:
            self.residual_down = nn.Linear(dim, 2 * cont_dim)  # mu, logvar
            self.residual_up = nn.Linear(cont_dim, dim)
        else:
            if self.variant == "vae":
                raise ValueError("variant 'vae' requires vae_dim > 0")
            self.residual_down = None
            self.residual_up = None

    @property
    def content_codebook_size(self) -> Optional[int]:
        if self.content_rvq is not None:
            return self.content_rvq.codebook_size
        return self.content_fsq.codebook_size if self.content_fsq else None

    @property
    def prosody_codebook_size(self) -> Optional[int]:
        return self.prosody_fsq.codebook_size if self.prosody_fsq else None

    def _continuous_channel(self, h: torch.Tensor):
        mu, logvar = self.residual_down(h).chunk(2, dim=-1)
        logvar = logvar.clamp(-8.0, 8.0)
        if self.training:
            residual = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            residual = mu
        # Per-frame KL (B, T): the caller masks padding before reducing.
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).mean(dim=-1)
        return residual, self.residual_up(residual), kl

    def forward(self, h: torch.Tensor) -> BottleneckOutput:
        """h: (B, T, dim) encoder output at the token rate."""
        c = p = None
        content_emb = prosody_emb = None
        q_loss = h.new_zeros(())
        if self.content_rvq is not None:
            c: RVQOutput = self.content_rvq(self.content_down(h))
            content_emb = self.content_up(c.quantized)
            q_loss = c.loss
        elif self.content_fsq is not None:
            c: FSQOutput = self.content_fsq(self.content_down(h))
            content_emb = self.content_up(c.quantized)
        if self.prosody_fsq is not None:
            p: FSQOutput = self.prosody_fsq(self.prosody_down(h))
            prosody_emb = self.prosody_up(p.quantized)

        residual = residual_emb = None
        kl = h.new_zeros(())
        if self.residual_down is not None:
            residual, residual_emb, kl = self._continuous_channel(h)

        return BottleneckOutput(
            content_emb=content_emb,
            prosody_emb=prosody_emb,
            residual_emb=residual_emb,
            content_indices=c.indices if c else None,
            prosody_indices=p.indices if p else None,
            content_continuous=c.continuous if c else None,
            prosody_continuous=p.continuous if p else None,
            residual=residual,
            content_quantized=c.quantized if c else None,
            prosody_quantized=p.quantized if p else None,
            kl_loss=kl,
            quantizer_loss=q_loss,
        )

    def embed_indices(self, content_indices: torch.Tensor,
                      prosody_indices: Optional[torch.Tensor] = None,
                      residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Rebuild the decoder condition purely from the discrete view.

        This is the detokenization path an index-predicting LM uses; the
        optional residual comes from a downstream continuous head (or is
        omitted, accepting the fidelity ceiling of the pure-discrete view).
        Not available for the "vae" variant (no discrete view - that's the point).
        """
        if self.content_rvq is not None:
            cond = self.content_up(self.content_rvq.indices_to_codes(content_indices))
        elif self.content_fsq is not None:
            cond = self.content_up(self.content_fsq.indices_to_codes(content_indices))
        else:
            raise RuntimeError("variant 'vae' has no discrete view to embed")
        if prosody_indices is not None:
            if self.prosody_fsq is None:
                raise RuntimeError(f"variant {self.variant!r} has no prosody stream")
            cond = cond + self.prosody_up(self.prosody_fsq.indices_to_codes(prosody_indices))
        if residual is not None and self.residual_up is not None:
            cond = cond + self.residual_up(residual)
        return cond
