"""scripts/padding_diagnostic.py on a tiny model: `fixed` equals encoding
alone, the non-causal model's legacy run differs, a causal encoder's tokens
do not."""

import dataclasses
import importlib.util
import sys
from pathlib import Path

import torch

from fact.config import tiny_config
from fact.data.synthetic import SyntheticSpeech, collate
from fact.model import FaCT

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("padding_diagnostic", ROOT / "scripts/padding_diagnostic.py")
pdg = importlib.util.module_from_spec(spec)
sys.modules["padding_diagnostic"] = pdg
spec.loader.exec_module(pdg)


def _setup(causal):
    cfg = tiny_config()
    cfg.encoder = dataclasses.replace(cfg.encoder, causal=causal)
    cfg.decoder = dataclasses.replace(cfg.decoder, causal=causal)
    torch.manual_seed(0)
    model = FaCT(cfg).eval()
    short = SyntheticSpeech(cfg, n_items=4, seconds=0.6, seed=1)
    long_ = SyntheticSpeech(cfg, n_items=4, seconds=1.5, seed=2)
    batches = [collate([short[i], long_[i], short[i + 1], long_[i + 1]], cfg) for i in (0, 2)]
    return model, batches


def test_noncausal_legacy_differs_and_fixed_equals_alone():
    model, batches = _setup(causal=False)
    rep = pdg.diagnose(model, batches, n_batches=2, alone_batches=2, device="cpu", ssl_fn=None)
    assert rep["n_batches"] == 2 and rep["padding"]["mean_frac"] > 0.2
    assert rep["fixed_vs_alone_content_agree"] == 1.0
    tok = rep["tokens_legacy_vs_fixed"]
    rec = rep["losses"]["recon"]
    assert tok["residual_rel_l2"] > 1e-4 or tok["content_agree"] < 1.0 or abs(rec["fixed_minus_legacy"]) > 1e-6
    assert set(rec) == {"legacy", "fixed", "fixed_minus_legacy", "ci95"}


def test_causal_encoder_tokens_unchanged():
    model, batches = _setup(causal=True)
    rep = pdg.diagnose(model, batches, n_batches=2, alone_batches=1, device="cpu", ssl_fn=None)
    assert rep["tokens_legacy_vs_fixed"]["content_agree"] == 1.0
    assert rep["fixed_vs_alone_content_agree"] == 1.0
