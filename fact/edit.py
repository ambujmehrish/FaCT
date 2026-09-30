"""Prosody-stream editing: the RQ2 interface claim made executable.

All operations follow the same semi-discrete pattern:

    indices -> continuous codes (read view) -> edit in code space
            -> nearest-lattice re-round (write view) -> valid indices

Edits are performed on the *prosody* stream only; content tokens are
untouched, so linguistic identity is preserved by construction (measured
as content-WER preservation in the RQ2 eval). Because `codes_to_indices`
rounds to the nearest lattice point, every edited state is a real discrete
state any index-predicting LM could itself have produced - there is no
out-of-vocabulary escape hatch.

Operations:
  swap:        prosody of B onto content of A (pure index exchange)
  interpolate: blend two prosody streams (alpha in [0, 1])
  neutralize:  flatten prosody to its utterance mean (monotone rendition)
  scale:       exaggerate (>1) or dampen (<1) deviation from the mean

Precedent note (verdict protocol): prosody exchange/flattening evals
follow the voice-conversion and FACodec/DisCo-Speech prosody-transfer
setups; the code-space arithmetic itself is enabled by the FSQ lattice.
"""

from __future__ import annotations

import torch

from .model import FaCT
from .modules.fsq import FSQ


def _prosody_fsq(model: FaCT) -> FSQ:
    fsq = model.bottleneck.prosody_fsq
    if fsq is None:
        raise RuntimeError(
            f"variant {model.bottleneck.variant!r} has no prosody stream to edit"
        )
    return fsq


def swap(content_indices: torch.Tensor, donor_prosody_indices: torch.Tensor,
         ) -> tuple[torch.Tensor, torch.Tensor]:
    """Content of A + prosody of B. Streams are aligned to the shorter
    utterance (prosody is time-aligned; cross-length transfer needs the
    donor cropped or stretched by the caller)."""
    t = min(content_indices.shape[1], donor_prosody_indices.shape[1])
    return content_indices[:, :t], donor_prosody_indices[:, :t]


@torch.no_grad()
def interpolate(model: FaCT, prosody_a: torch.Tensor, prosody_b: torch.Tensor,
                alpha: float) -> torch.Tensor:
    """Blend prosody streams: alpha=0 -> A, alpha=1 -> B."""
    fsq = _prosody_fsq(model)
    t = min(prosody_a.shape[1], prosody_b.shape[1])
    ca = fsq.indices_to_codes(prosody_a[:, :t])
    cb = fsq.indices_to_codes(prosody_b[:, :t])
    return fsq.codes_to_indices((1 - alpha) * ca + alpha * cb)


@torch.no_grad()
def neutralize(model: FaCT, prosody_indices: torch.Tensor) -> torch.Tensor:
    """Flatten prosody variation to the utterance mean (per item)."""
    return scale(model, prosody_indices, 0.0)


@torch.no_grad()
def scale(model: FaCT, prosody_indices: torch.Tensor, factor: float) -> torch.Tensor:
    """Scale deviation from the utterance-mean prosody code.

    factor 0 = flat, 1 = identity (up to lattice rounding), >1 = exaggerated.
    """
    fsq = _prosody_fsq(model)
    codes = fsq.indices_to_codes(prosody_indices)
    mean = codes.mean(dim=1, keepdim=True)
    return fsq.codes_to_indices(mean + factor * (codes - mean))
