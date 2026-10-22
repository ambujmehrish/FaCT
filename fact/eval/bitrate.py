"""Nominal FaCT bitrate per stream at the token rate."""

from __future__ import annotations

import math


def fact_bitrate(model) -> dict:
    cfg = model.cfg
    tok_rate = cfg.audio.sample_rate / (cfg.audio.hop_length * cfg.encoder.frame_stack)
    bn = model.bottleneck
    out = {"token_rate_hz": tok_rate}
    if bn.content_codebook_size:
        out["content_bps"] = math.log2(bn.content_codebook_size) * tok_rate
    if bn.prosody_codebook_size:
        out["prosody_bps"] = math.log2(bn.prosody_codebook_size) * tok_rate / bn.prosody_rate_divisor
    if bn.residual_down is not None:
        # Capped channel: Shannon capacity bound. KL channel: no bound.
        out["residual_bps"] = (bn.residual_capacity / math.log(2) * tok_rate
                               if bn.residual_capacity > 0 else None)
    out["discrete_bps"] = out.get("content_bps", 0.0) + out.get("prosody_bps", 0.0)
    return out
