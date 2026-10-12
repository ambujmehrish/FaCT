"""Prosody editing, lattice write-path, multi-rate prosody, tripwires."""

import json
import math

import pytest
import torch

from fact import edit
from fact.config import tiny_config
from fact.data.synthetic import SyntheticSpeech, collate
from fact.model import FaCT
from fact.modules.fsq import FSQ


def _batch(cfg, n=2, seconds=1.0):
    ds = SyntheticSpeech(cfg, n_items=n, seconds=seconds)
    return collate([ds[i] for i in range(n)], cfg)


# --------------------------------------------------- lattice write-path (FSQ)

def test_codes_to_indices_inverts_indices_to_codes():
    for levels in ([5, 5, 5], [8, 8, 4], [4, 4], [3, 2]):
        fsq = FSQ(levels)
        idx = torch.arange(fsq.codebook_size)
        assert torch.equal(fsq.codes_to_indices(fsq.indices_to_codes(idx)), idx), levels


def test_codes_to_indices_rounds_to_nearest_lattice():
    fsq = FSQ([5, 5])
    idx = torch.tensor([7, 13])
    codes = fsq.indices_to_codes(idx)
    noisy = codes + 0.15  # less than half a cell (cell = 0.5 in [-1,1])
    assert torch.equal(fsq.codes_to_indices(noisy), idx)
    # Out-of-range vectors clamp to valid lattice ids.
    wild = fsq.codes_to_indices(torch.randn(100, 2) * 5)
    assert wild.min() >= 0 and wild.max() < fsq.codebook_size


# -------------------------------------------------------------- prosody edits

def test_edit_scale_identity_and_flat():
    cfg = tiny_config()
    model = FaCT(cfg).eval()
    toks = model.tokenize(mel=_batch(cfg).mel)
    p = toks.prosody_indices
    assert torch.equal(edit.scale(model, p, 1.0), p)  # identity at factor 1
    flat = edit.neutralize(model, p)
    # Flattened: (near-)constant stream per item, and still valid indices.
    assert flat.shape == p.shape
    for row in flat:
        assert row.unique().numel() <= 2  # mean rounds to <=2 neighboring cells
    assert flat.max() < model.bottleneck.prosody_codebook_size


def test_edit_interpolate_endpoints_and_validity():
    cfg = tiny_config()
    model = FaCT(cfg).eval()
    b = _batch(cfg, n=2)
    toks = model.tokenize(mel=b.mel)
    pa, pb = toks.prosody_indices[:1], toks.prosody_indices[1:2]
    assert torch.equal(edit.interpolate(model, pa, pb, 0.0), pa)
    assert torch.equal(edit.interpolate(model, pa, pb, 1.0), pb)
    mid = edit.interpolate(model, pa, pb, 0.5)
    assert mid.min() >= 0 and mid.max() < model.bottleneck.prosody_codebook_size


def test_edited_streams_decode():
    cfg = tiny_config()
    model = FaCT(cfg).eval()
    b = _batch(cfg, n=2)
    toks = model.tokenize(mel=b.mel)
    c, p = edit.swap(toks.content_indices[:1], toks.prosody_indices[1:2])
    mel = model.detokenize(c, edit.scale(model, p, 1.5), ref_mel=b.mel[:1], nfe=1)
    assert torch.isfinite(mel).all()


def test_edit_requires_prosody_stream():
    cfg = tiny_config()
    cfg.bottleneck.variant = "vae"
    cfg.bottleneck.vae_dim = 9
    model = FaCT(cfg)
    with pytest.raises(RuntimeError):
        edit.scale(model, torch.zeros(1, 4, dtype=torch.long), 0.5)


# --------------------------------------------------------- multi-rate prosody

def test_multirate_shapes_and_training():
    cfg = tiny_config()
    cfg.bottleneck.prosody_rate_divisor = 2
    model = FaCT(cfg)
    batch = _batch(cfg)
    model.eval()
    toks = model.tokenize(mel=batch.mel)
    t_c = toks.content_indices.shape[1]
    assert toks.prosody_indices.shape[1] == math.ceil(t_c / 2)
    # Low-rate prosody indices decode against full-rate content.
    mel = model.detokenize(toks.content_indices, toks.prosody_indices,
                           ref_mel=batch.mel, residual=toks.residual, nfe=1)
    assert torch.isfinite(mel).all()
    model.train()
    logs = model.training_step(batch)
    for k, v in logs.items():
        assert torch.isfinite(v), k
    logs["loss"].backward()


def test_multirate_edits_still_work():
    cfg = tiny_config()
    cfg.bottleneck.prosody_rate_divisor = 2
    model = FaCT(cfg).eval()
    toks = model.tokenize(mel=_batch(cfg).mel)
    flat = edit.neutralize(model, toks.prosody_indices)
    assert flat.shape == toks.prosody_indices.shape


def test_multirate_config_loads():
    from fact.config import FaCTConfig
    cfg = FaCTConfig.from_yaml("configs/ablation_prosody_slow.yaml")
    assert cfg.bottleneck.prosody_rate_divisor == 2


# ------------------------------------------------------------------ tripwires

def test_tripwires_end_to_end(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("tripwires", Path("scripts/tripwires.py"))
    tw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tw)

    cfg = tiny_config()
    # Shards + checkpoint on disk, like the real gate review.
    ds = SyntheticSpeech(cfg, n_items=6, seconds=1.0)
    items = [{"mel": ds[i]["mel"].to(torch.float16), "f0": ds[i]["f0"],
              "text": ds[i]["text"]} for i in range(6)]
    torch.save(items, tmp_path / "shard_000000.pt")
    model = FaCT(cfg)
    ckpt_path = tmp_path / "step.pt"
    torch.save({"model": model.state_dict(), "step": 1,
                "config": cfg.to_dict()}, ckpt_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg.save_yaml(str(cfg_path))

    import sys
    argv = sys.argv
    sys.argv = ["tripwires.py", "--config", str(cfg_path), "--ckpt", str(ckpt_path),
                "--shards", str(tmp_path), "--out", str(tmp_path / "report.json"),
                "--n-utts", "6", "--device", "cpu"]
    try:
        tw.main()
    finally:
        sys.argv = argv
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["content"]["codebook_size"] == 125
    assert 0 < report["content"]["utilization"] <= 1
    assert "prosody" in report and "perplexity" in report["prosody"]
    z = report["channel_zeroing"]
    assert set(z["delta_vs_full"]) == {"content", "prosody", "residual"}
    assert "content_to_f0_acc" in report["probes"]
    assert isinstance(report["flags"], dict)
