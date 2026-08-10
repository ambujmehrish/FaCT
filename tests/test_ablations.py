"""RVQ quantizer-swap and non-causal ablations + offline-cache guard."""

import math

import pytest
import torch

from fact.config import BottleneckConfig, FaCTConfig, tiny_config
from fact.data.synthetic import SyntheticSpeech, collate
from fact.model import FaCT
from fact.modules.rvq import ResidualVQ


def _rvq_cfg():
    cfg = tiny_config()
    cfg.bottleneck.variant = "rvq"
    cfg.bottleneck.rvq_codebook_sizes = (11, 11)  # tiny: 121 ~ content 125
    cfg.bottleneck.rvq_dim = 4
    return cfg


def _batch(cfg, n=2):
    ds = SyntheticSpeech(cfg, n_items=n, seconds=1.0)
    return collate([ds[i] for i in range(n)], cfg)


# ------------------------------------------------------------------ RVQ module

def test_rvq_flat_index_roundtrip():
    torch.manual_seed(0)
    rvq = ResidualVQ(dim=4, codebook_sizes=[7, 5]).eval()
    z = torch.randn(3, 20, 4)
    out = rvq(z)
    assert out.indices.shape == (3, 20)
    assert out.indices.min() >= 0 and out.indices.max() < 35
    recon = rvq.indices_to_codes(out.indices)
    assert torch.allclose(recon, out.quantized, atol=1e-5)


def test_rvq_ste_gradient_and_commitment():
    torch.manual_seed(0)
    rvq = ResidualVQ(dim=4, codebook_sizes=[7, 5]).train()
    z = torch.randn(2, 10, 4, requires_grad=True)
    out = rvq(z)
    (out.quantized.sum() + out.loss).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert float(out.loss) > 0


def test_rvq_ema_moves_codebooks():
    torch.manual_seed(0)
    rvq = ResidualVQ(dim=4, codebook_sizes=[7, 5], decay=0.5).train()
    before = rvq._embed(0).clone()
    for _ in range(5):
        rvq(torch.randn(2, 50, 4) * 3)
    assert not torch.allclose(before, rvq._embed(0))


def test_rvq_matched_bits_default():
    b = BottleneckConfig()
    rvq_bits = math.log2(math.prod(b.rvq_codebook_sizes))
    fsq_bits = math.log2(math.prod(b.content_levels))
    assert abs(rvq_bits - fsq_bits) < 0.1  # 2^13.02 vs 2^13


# --------------------------------------------------------------- RVQ variant

def test_rvq_variant_trains_with_full_factorization():
    cfg = _rvq_cfg()
    model = FaCT(cfg)
    # RVQ keeps the factorization: prosody supervision + leakage present.
    assert model.prosody_predictor is not None and model.leakage is not None
    batch = _batch(cfg)
    logs = model.training_step(batch, stage="a")
    assert "commit" in logs
    for k, v in logs.items():
        assert torch.isfinite(v), k
    logs["loss"].backward()
    grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert grads


def test_rvq_variant_tokenize_detokenize():
    cfg = _rvq_cfg()
    model = FaCT(cfg).eval()
    batch = _batch(cfg)
    toks = model.tokenize(mel=batch.mel)
    assert toks.content_indices.max() < model.bottleneck.content_codebook_size
    assert toks.prosody_indices is not None  # factorization intact
    mel = model.detokenize(toks.content_indices, toks.prosody_indices,
                           ref_mel=batch.mel, residual=toks.residual, nfe=1)
    assert torch.isfinite(mel).all()


def test_ablation_yaml_configs_load():
    cfg = FaCTConfig.from_yaml("configs/ablation_rvq.yaml")
    assert cfg.bottleneck.variant == "rvq"
    cfg = FaCTConfig.from_yaml("configs/ablation_noncausal.yaml")
    assert not cfg.encoder.causal and not cfg.decoder.causal


# ---------------------------------------------------------------- non-causal

def test_noncausal_encoder_sees_future():
    cfg = tiny_config()
    cfg.encoder.causal = False
    model = FaCT(cfg).eval()
    torch.manual_seed(0)
    mel = torch.randn(1, 32, cfg.audio.n_mels)
    mel2 = mel.clone()
    mel2[:, 16:] += 10.0
    with torch.no_grad():
        h1 = model.encoder(mel)
        h2 = model.encoder(mel2)
    # Bidirectional: early outputs MUST change when the future changes.
    assert not torch.allclose(h1[:, :4], h2[:, :4], atol=1e-3)


def _dezero(module):
    """adaLN gates and the out proj are zero-initialized (DiT-style), which
    makes an untrained decoder output constants - randomize them so
    causality assertions are non-vacuous."""
    with torch.no_grad():
        for p in module.parameters():
            if p.abs().sum() == 0:
                torch.nn.init.normal_(p, std=0.1)


def test_noncausal_decoder_sees_future():
    cfg = tiny_config()
    cfg.decoder.causal = False
    model = FaCT(cfg).eval()
    _dezero(model.decoder)
    torch.manual_seed(0)
    t_tok = 8
    x_t = torch.randn(1, t_tok * cfg.encoder.frame_stack, cfg.audio.n_mels)
    cond = torch.randn(1, t_tok, cfg.encoder.dim)
    spk = torch.randn(1, cfg.speaker.dim)
    cond2 = cond.clone()
    cond2[:, 4:] += 10.0
    with torch.no_grad():
        v1 = model.decoder(x_t, torch.tensor([0.5]), cond, spk)
        v2 = model.decoder(x_t, torch.tensor([0.5]), cond2, spk)
    assert not torch.allclose(v1[:, :16], v2[:, :16], atol=1e-3)


def test_noncausal_training_step():
    cfg = tiny_config()
    cfg.encoder.causal = False
    cfg.decoder.causal = False
    logs = FaCT(cfg).training_step(_batch(cfg))
    assert torch.isfinite(logs["loss"])


# ------------------------------------------------------------- cache guard

def test_download_cache_refuses_home(monkeypatch, tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "download_checkpoints", Path("scripts/download_checkpoints.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    with pytest.raises(SystemExit, match="HOME"):
        mod.resolve_cache_dir(str(tmp_path / "weights"), allow_home=False)
    # Explicit override works; outside $HOME works.
    assert mod.resolve_cache_dir(str(tmp_path / "weights"), allow_home=True).exists()
    monkeypatch.delenv("FACT_CACHE", raising=False)
    with pytest.raises(SystemExit, match="FACT_CACHE"):
        mod.resolve_cache_dir(None, allow_home=False)