"""Eval metrics must ignore batch padding (sweep + tripwire measurements)."""

import torch

from fact.config import tiny_config
from fact.data.synthetic import SyntheticSpeech, collate
from fact.eval.grounding import (codebook_stats, greedy_byte_cer, token_lengths,
                                 valid_indices)
from fact.model import FaCT


def test_cer_and_codes_unchanged_by_padding():
    cfg = tiny_config()
    torch.manual_seed(0)
    model = FaCT(cfg).eval()
    short = SyntheticSpeech(cfg, n_items=1, seconds=1.0)[0]
    long = SyntheticSpeech(cfg, n_items=1, seconds=3.0, seed=7)[0]
    alone = collate([short], cfg)
    padded = collate([short, long], cfg)
    # Score only the short item in both cases.
    padded_short = type(padded)(**{**padded.__dict__})
    padded_short.text_lengths = padded.text_lengths.clone()
    padded_short.text_lengths[1] = 0
    assert greedy_byte_cer(model, [alone], "cpu") == greedy_byte_cer(model, [padded_short], "cpu")

    s = cfg.encoder.frame_stack
    with torch.no_grad():
        a = model.encode_mel(alone.mel).content_indices
        p = model.encode_mel(padded.mel).content_indices
    a_idx = valid_indices(a, token_lengths(alone, a.shape[1], s))
    p_idx = valid_indices(p, token_lengths(padded, p.shape[1], s))
    assert torch.equal(p_idx[: a_idx.numel()], a_idx)
    assert p_idx.numel() == a_idx.numel() + int(padded.mel_lengths[1]) // s


def test_valid_indices_slow_prosody_rate():
    idx = torch.arange(10).view(2, 5)
    out = valid_indices(idx, torch.tensor([7, 3]), rate_divisor=2)
    assert out.tolist() == [0, 1, 2, 3, 5, 6]  # ceil(7/2)=4, ceil(3/2)=2


def test_codebook_stats_reports_token_count():
    st = codebook_stats(torch.tensor([0, 0, 1, 3]), 8)
    assert st["tokens"] == 4 and st["codes_used"] == 3
    assert abs(st["utilization"] - 3 / 8) < 1e-9


# ------------------------------------------------- capacity-capped residual

def test_capped_residual_channel_bound_and_paths():
    import math
    cfg = tiny_config()
    cfg.bottleneck.residual_capacity = 2.0  # nats / frame
    model = FaCT(cfg)
    bn = model.bottleneck
    d = cfg.bottleneck.residual_dim
    sigma = bn.residual_noise_std
    # d/2 ln(1 + 1/sigma^2) == capacity (AWGN bound, unit amplitude).
    assert abs(d / 2 * math.log1p(1 / sigma**2) - 2.0) < 1e-9
    batch = collate([SyntheticSpeech(cfg, n_items=2, seconds=1.0)[i] for i in range(2)], cfg)
    logs = model.training_step(batch)
    assert float(logs["kl"]) == 0.0 and torch.isfinite(logs["loss"])
    model.eval()
    with torch.no_grad():
        out = model.encode_mel(batch.mel)
        assert out.residual.abs().max() <= 1.0
        # Noise stays on at inference: two reads of the same message differ.
        e1 = bn.embed_indices(out.content_indices, out.prosody_indices, out.residual)
        e2 = bn.embed_indices(out.content_indices, out.prosody_indices, out.residual)
        assert not torch.allclose(e1, e2)


def test_sweep_runner_param_path():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("sweep", Path("scripts/sweep_lambda_ctc.py"))
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)
    cfg = tiny_config()
    sweep.PARAM = "bottleneck.residual_capacity"
    arm = sweep.arm_config(cfg, 4.0)
    assert arm.bottleneck.residual_capacity == 4.0 and cfg.bottleneck.residual_capacity == 0.0
    assert sweep.arm_dir(Path("o"), 4.0) == Path("o/residual_capacity_4")
    sweep.PARAM = "bottleneck.residual_dim"
    assert sweep.arm_config(cfg, 0).bottleneck.residual_dim == 0
    sweep.PARAM = "ctc.weight"
    assert sweep.arm_dir(Path("o"), 3.0) == Path("o/lam_3")
