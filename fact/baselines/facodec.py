"""FACodec (NaturalSpeech 3) - the factorized-codec reference baseline.

The right disentanglement comparison for the leakage matrix: FVQ streams
for content/prosody/detail + timbre, GRL supervision, NON-causal - i.e.
factorization without our causality/semi-discreteness/interface claims.
Public checkpoint: HF `amphion/naturalspeech3_facodec`.

STUB: the numerical wrapper is deferred until its runtime dependencies
are validated on Leonardo (PIPELINE_ANALYSIS P2) - Amphion's ns3_codec
stack is heavy and cannot be verified in the dev sandbox, and shipping an
unverified integration would violate the verdict protocol. Building it:

    pip install amphion  # or clone github.com/open-mmlab/Amphion
    # wrap FACodecEncoder/FACodecDecoder: expose per-stream indices via
    # tokenize() (content/prosody/detail streams separately - that is the
    # point of comparing against it) and reconstruct(); read frame rate
    # and codebook sizes from the loaded model per base.py conventions.
"""

from __future__ import annotations

from .base import register


@register("facodec")
def _build(device: str = "cuda", **kwargs):
    raise NotImplementedError(
        "FACodec wrapper is a stub (see module docstring): validate the "
        "Amphion ns3_codec runtime on the cluster, then implement "
        "tokenize()/reconstruct() per fact/baselines/base.py. Tracked in "
        "docs/PIPELINE_ANALYSIS.md P2."
    )
