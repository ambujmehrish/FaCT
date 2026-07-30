"""Matched internal baselines: single_fsq (no factorization) and vae
(pure continuous). These are the fairness baselines of RQ1-RQ3."""

import math

import pytest
import torch

from fact.config import BottleneckConfig, FaCTConfig, tiny_config
from fact.data.synthetic import SyntheticSpeech, collate
from fact.model import FaCT


def _cfg(variant):
    cfg = tiny_config()
    cfg.bottleneck.variant = variant
    cfg.bottleneck.single_levels = (5, 5, 5, 4, 4)  # tiny: content+prosody union
    cfg.bottleneck.vae_dim = 9
    return cfg


def _batch(cfg, n=2):
    ds = SyntheticSpeech(cfg, n_items=n, seconds=1.0)
    return collate([ds[i] for i in range(n)], cfg)


def test_matched_bits_default_config():
    """Default single_levels = union of content and prosody lattices: exact bit match."""
    b = BottleneckConfig()
    fact_bits = math.log2(math.prod(b.content_levels) * math.prod(b.prosody_levels))
    single_bits = math.log2(math.prod(b.single_levels))
    assert abs(fact_bits - single_bits) < 1e-9
    assert len(b.single_levels) == len(b.content_levels) + len(b.prosody_levels)
    # And matched continuous dims for the VAE baseline.
    assert b.vae_dim == len(b.content_levels) + len(b.prosody_levels) + b.residual_dim


def test_single_fsq_variant():
    cfg = _cfg("single_fsq")
    model = FaCT(cfg)
    assert model.prosody_predictor is None and model.leakage is None
    batch = _batch(cfg)
    logs = model.training_step(batch, stage="a")
    assert torch.isfinite(logs["loss"])
    assert "f0" not in logs and "leak_c2f" not in logs
    logs["loss"].backward()

    model.eval()
    toks = model.tokenize(mel=batch.mel)
    assert toks.prosody_indices is None
    assert toks.content_indices.max() < model.bottleneck.content_codebook_size
    assert [n for n, _ in toks.index_streams()] == ["content"]
    mel = model.detokenize(toks.content_indices, None, ref_mel=batch.mel,
                           residual=toks.residual, nfe=1)
    assert torch.isfinite(mel).all()


def test_vae_variant():
    cfg = _cfg("vae")
    model = FaCT(cfg)
    batch = _batch(cfg)
    logs = model.training_step(batch, stage="a")
    assert torch.isfinite(logs["loss"])
    assert float(logs["kl"]) > 0  # the VAE latent is KL-regularized
    logs["loss"].backward()

    model.eval()
    toks = model.tokenize(mel=batch.mel)
    assert toks.content_indices is None and toks.prosody_indices is None
    assert toks.residual.shape[-1] == cfg.bottleneck.vae_dim
    assert toks.index_streams() == []  # no discrete view - that's the point
    with pytest.raises(RuntimeError):
        model.bottleneck.embed_indices(torch.zeros(1, 4, dtype=torch.long))
    # Continuous round-trip still works.
    mel = model.reconstruct(batch.mel, nfe=1)
    assert torch.isfinite(mel).all()


def test_vae_stage_b_freeze():
    cfg = _cfg("vae")
    model = FaCT(cfg)
    model.freeze_tokenizer()
    logs = model.training_step(_batch(cfg), stage="b")
    assert torch.isfinite(logs["loss"])


def test_unknown_variant_rejected():
    cfg = _cfg("factorized")
    cfg.bottleneck.variant = "bogus"
    with pytest.raises(ValueError):
        FaCT(cfg)


def test_baseline_yaml_configs_load():
    for path in ("configs/baseline_single_fsq.yaml", "configs/baseline_vae.yaml"):
        cfg = FaCTConfig.from_yaml(path)
        assert cfg.bottleneck.variant in ("single_fsq", "vae")
