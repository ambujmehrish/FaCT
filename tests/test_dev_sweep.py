"""Sampling controls in the FaCT eval: temperature / CFG reach the decoder,
sample 1 of K equals the K=1 result, diagnostics are reported."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from fact.config import AudioConfig, tiny_config
from fact.model import FaCT


def _run(tmp_path, monkeypatch, *extra):
    import soundfile as sf
    cfg = tiny_config()
    cfg.audio = AudioConfig()
    cfg.bottleneck.residual_capacity = 2.0
    ck = tmp_path / "ck.pt"
    if not ck.exists():
        torch.manual_seed(0)
        torch.save({"model": FaCT(cfg).state_dict(), "step": 1, "config": cfg.to_dict()}, ck)
        d = tmp_path / "w"
        d.mkdir()
        x = 0.1 * np.random.default_rng(0).standard_normal(16_000)
        x[:4_000] *= 1e-4  # a quiet stretch -> silent frames exist
        sf.write(d / "u0.wav", x.astype("float32"), 16_000)
    out = tmp_path / f"r{len(list(tmp_path.glob('r*.json')))}.json"
    monkeypatch.setattr(sys, "argv", ["x", "--ckpt", str(ck), "--eval-dir", str(tmp_path / "w"),
                                      "--out", str(out), "--nfe", "2", "--device", "cpu",
                                      "--views", "full", *extra])
    spec = importlib.util.spec_from_file_location("e", Path("scripts/eval_fact_recon.py"))
    e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e)
    e.main()
    return json.loads(out.read_text())


def test_sampling_controls_and_diagnostics(tmp_path, monkeypatch):
    a = _run(tmp_path, monkeypatch)
    k = _run(tmp_path, monkeypatch, "--n-samples", "3")
    t = _run(tmp_path, monkeypatch, "--temperature", "0.5")
    u = "u0.wav"
    assert k["per_utterance"][u]["full/mel_l1_band"] == a["per_utterance"][u]["full/mel_l1_band"]
    assert "full/mel_l1_band_mean3" in k["per_utterance"][u]
    assert t["per_utterance"][u]["full/mel_l1_band"] != a["per_utterance"][u]["full/mel_l1_band"]
    m = a["per_utterance"][u]
    assert 0 < m["full/active_frac"] < 1 and "full/mel_l1_band_silent" in m
    assert t["temperature"] == 0.5 and k["n_samples"] == 3
