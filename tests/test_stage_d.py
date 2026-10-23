"""Stage D (#2): waveform decoder on frozen FaCT tokens."""

import sys

import numpy as np
import torch

from fact.config import tiny_config
from fact.model import FaCT


def _corpus(tmp_path, n=4, sr=16_000):
    import soundfile as sf
    rng = np.random.default_rng(0)
    lines = []
    for k in range(n):
        t = np.arange(sr) / sr
        x = 0.3 * np.sin(2 * np.pi * (150 + 30 * k) * t) + 0.01 * rng.standard_normal(sr)
        p = tmp_path / f"u{k}.wav"
        sf.write(p, x.astype("float32"), sr)
        lines.append(f"{p}\ttext\n")
    m = tmp_path / "m.tsv"
    m.write_text("".join(lines))
    return m


def test_stage_d_trains_frozen_fact_and_wraps(tmp_path, monkeypatch):
    from fact.baselines import base as registry
    from fact.train import stage_d_wave_decoder as sd
    cfg = tiny_config()
    cfg.bottleneck.residual_capacity = 2.0
    cfg.bottleneck.residual_dropout = 0.5
    torch.manual_seed(0)
    fact = FaCT(cfg)
    fck = tmp_path / "fact.pt"
    torch.save({"model": fact.state_dict(), "step": 1, "config": cfg.to_dict()}, fck)
    before = {k: v.clone() for k, v in fact.state_dict().items()}
    m = _corpus(tmp_path)
    ck = tmp_path / "ck"
    monkeypatch.setattr(sys, "argv", ["x", "--tiny", "--fact-ckpt", str(fck), "--manifest", str(m),
                                      "--dev-manifest", str(m), "--ckpt-dir", str(ck),
                                      "--device", "cpu", "--steps", "2"])
    sd.main()
    saved = torch.load(ck / "step_0000002.pt", weights_only=True)
    assert saved["residual_drop"] == 0.5 and saved["in_dim"] == sd.input_dim(fact)
    # The FaCT checkpoint itself is untouched (frozen, never saved).
    after = torch.load(fck, weights_only=True)["model"]
    assert all(torch.equal(before[k], after[k]) for k in before)
    wav = 0.1 * torch.randn(1, 16 * 320)
    for view in ("full", "discrete_only"):
        tok = registry.build("fact_wave", device="cpu", decoder=str(ck / "step_0000002.pt"), view=view)
        out = tok.reconstruct(wav)
        assert out.shape == (1, 15 * 320) and torch.isfinite(out).all()
    # Deterministic apart from the capped residual's channel noise.
    torch.manual_seed(3)
    a = registry.build("fact_wave", device="cpu", decoder=str(ck / "step_0000002.pt"),
                       view="discrete_only").reconstruct(wav)
    torch.manual_seed(4)
    b = registry.build("fact_wave", device="cpu", decoder=str(ck / "step_0000002.pt"),
                       view="discrete_only").reconstruct(wav)
    assert torch.allclose(a, b)


def test_fact_vocoded_passes_temperature(tmp_path, monkeypatch):
    import fact.baselines.fact_vocoded as fv
    seen = {}

    class Stub:
        def __init__(self, *a, **k):
            seen.update(k)
    monkeypatch.setattr(fv, "FaCTVocoded", Stub)
    fv._build_fact(device="cpu", ckpt="c", vocoder="v", temperature="0.667", cfg_scale="1.5")
    assert seen["temperature"] == "0.667" and seen["cfg_scale"] == "1.5"
