"""Streaming-valid speaker references: keys, partners, and the eval option."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from fact.config import AudioConfig, tiny_config
from fact.eval.eval_set import reference_partners, speaker_key
from fact.model import FaCT


def test_speaker_key():
    assert speaker_key("0000_6930-75918-0000.wav") == "6930"
    assert speaker_key("EN_B00082_S04956_W000007") == "EN_B00082_S04956"
    with pytest.raises(ValueError):
        speaker_key("u0.wav")


def test_reference_partners_same_speaker_other_utterance():
    keys = ["0000_10-1-0.wav", "0001_10-1-1.wav", "0002_10-2-0.wav", "0003_20-1-0.wav",
            "0004_20-1-1.wav"]
    p = reference_partners(keys)
    for k, v in p.items():
        assert v != k and speaker_key(v) == speaker_key(k)
    with pytest.raises(ValueError, match="one utterance"):
        reference_partners(keys + ["0005_30-1-0.wav"])


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(f"scripts/{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("ref", ["full", "prefix:0.5", "other"])
def test_eval_fact_recon_spk_ref(tmp_path, monkeypatch, ref):
    import soundfile as sf
    cfg = tiny_config()
    cfg.audio = AudioConfig()
    cfg.bottleneck.residual_capacity = 2.0
    ckpt = tmp_path / "step.pt"
    torch.save({"model": FaCT(cfg).state_dict(), "step": 1, "config": cfg.to_dict()}, ckpt)
    d = tmp_path / "wavs"
    d.mkdir()
    rng = np.random.default_rng(0)
    for name in ("0000_10-1-0.wav", "0001_10-1-1.wav"):
        sf.write(d / name, (0.1 * rng.standard_normal(16_000)).astype("float32"), 16_000)
    out = tmp_path / "r.json"
    monkeypatch.setattr(sys, "argv", ["x", "--ckpt", str(ckpt), "--eval-dir", str(d),
                                      "--out", str(out), "--nfe", "2", "--device", "cpu",
                                      "--views", "full", "--spk-ref", ref])
    _load("eval_fact_recon").main()
    r = json.loads(out.read_text())
    assert r["spk_ref"] == ref and r["views"]["full"]["mel_l1"]["n"] == 2


def test_eval_fact_recon_rejects_bad_spk_ref(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--ckpt", "missing.pt", "--eval-dir", str(tmp_path),
                                      "--out", str(tmp_path / "r.json"), "--spk-ref", "prefix"])
    with pytest.raises(BaseException):
        _load("eval_fact_recon").main()
