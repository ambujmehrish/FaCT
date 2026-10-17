"""FaCT mel-domain reconstruction eval and stage-A sampling."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from fact.config import AudioConfig, tiny_config
from fact.model import FaCT


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(f"scripts/{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stage_a_sampling_uses_plain_flow_matching():
    cfg = tiny_config()
    torch.manual_seed(0)
    dec = FaCT(cfg).decoder.eval()
    cond = torch.randn(1, 3, dec.cond_proj.in_features)
    spk = torch.randn(1, dec.spk_proj.in_features)
    torch.manual_seed(1)
    got = dec.sample(cond, spk, nfe=4, shortcut=False)
    torch.manual_seed(1)
    x = torch.randn_like(got)
    for i in range(4):
        t = torch.full((1,), i / 4)
        x = x + 0.25 * dec(x, t, cond, spk)  # d=None -> d=0, the stage-A path
    assert torch.allclose(got, x, atol=1e-6)


def test_eval_fact_recon_end_to_end(tmp_path, monkeypatch):
    import soundfile as sf
    cfg = tiny_config()
    cfg.audio = AudioConfig()  # the baselines' mel space (24 kHz, 100 mels)
    cfg.bottleneck.residual_capacity = 2.0
    model = FaCT(cfg)
    ckpt = tmp_path / "step.pt"
    torch.save({"model": model.state_dict(), "step": 7, "config": cfg.to_dict()}, ckpt)
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    rng = np.random.default_rng(0)
    for k in range(2):
        sf.write(wav_dir / f"u{k}.wav", (0.1 * rng.standard_normal(16_000)).astype("float32"), 16_000)
    out = tmp_path / "r.json"
    ev = _load("eval_fact_recon")
    monkeypatch.setattr(sys, "argv", ["eval_fact_recon.py", "--ckpt", str(ckpt),
                                      "--eval-dir", str(wav_dir), "--out", str(out),
                                      "--nfe", "2", "--device", "cpu"])
    ev.main()
    r = json.loads(out.read_text())
    assert r["step"] == 7 and r["n_utterances"] == 2
    for view in ("full", "discrete_only"):
        assert r["views"][view]["mel_l1"]["n"] == 2
        assert math.isfinite(r["views"][view]["mel_l1"]["mean"])
    assert set(r["per_utterance"]) == {"u0.wav", "u1.wav"}
    assert "discrete_only/mel_l1" in r["per_utterance"]["u0.wav"]
    br = r["bitrate"]
    assert abs(br["content_bps"] - math.log2(125) * 12.5) < 1e-9
    assert abs(br["residual_bps"] - 2.0 / math.log(2) * 12.5) < 1e-9


def test_eval_fact_recon_refuses_other_mel_space():
    ev = _load("eval_fact_recon")
    with pytest.raises(SystemExit):
        ev.check_mel_space(tiny_config().audio)


def test_baseline_spec_parsing():
    eb = _load("eval_baselines")
    assert eb.parse_spec("mimi:n_streams=2") == ("mimi", {"n_streams": 2})
    assert eb.parse_spec("dac") == ("dac", {})
    assert eb.spec_label("dac:n_quantizers=4") == "dac_n_quantizers4"
    assert eb.spec_label("mimi") == "mimi"
