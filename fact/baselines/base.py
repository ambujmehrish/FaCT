"""Uniform interface for external public-checkpoint baselines (Tier 2).

Every baseline exposes:
  - reconstruct(wav) -> wav  (full encode->decode round trip, on `device`)
  - tokenize(wav) -> list of (B, T) index streams, or None (continuous)
  - metadata read from the checkpoint itself (frame rate, bitrate, vocab)
    so no constants are hardcoded and numbers in tables are verifiable.

Checkpoints must be pre-downloaded on a login node (CINECA compute nodes
have no internet): see scripts/download_checkpoints.py and docs/CINECA.md.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol

import torch


class BaselineTokenizer(Protocol):
    name: str
    sample_rate: int
    frame_rate_hz: float
    discrete_bps: Optional[float]  # None for continuous representations
    vocab_sizes: Optional[list[int]]

    def reconstruct(self, wav: torch.Tensor) -> torch.Tensor: ...
    def tokenize(self, wav: torch.Tensor) -> Optional[list[torch.Tensor]]: ...


_REGISTRY: dict[str, Callable[..., "BaselineTokenizer"]] = {}


def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def available() -> list[str]:
    return sorted(_REGISTRY)


def build(name: str, device: str = "cuda", **kwargs) -> "BaselineTokenizer":
    if name not in _REGISTRY:
        raise KeyError(f"Unknown baseline {name!r}; available: {available()}")
    return _REGISTRY[name](device=device, **kwargs)


# Import wrapper modules so their @register decorators run. Wrappers guard
# their heavy imports internally (build() raises a helpful ImportError).
def _autoload() -> None:
    from . import dac_codec, mimi  # noqa: F401
    try:
        from . import xcodec2  # noqa: F401
    except Exception:
        pass


_autoload()
