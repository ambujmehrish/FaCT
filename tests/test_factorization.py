"""Factorization evaluation: WORLD manipulations really move the pitch, the
metrics are right, and the end-to-end script + shard merge run on a tiny
model with a tiny stage-D decoder."""

import json
import sys

import numpy as np
import pytest
import torch

from fact.data.prosody_targets import extract_f0_pyworld
from fact.eval import factorization as fz


def _voiced(sr=16_000, sec=1.2, f=140.0, seed=0):
    """Harmonic signal with vibrato (WORLD finds a clean F0)."""
    t = np.arange(int(sr * sec)) / sr
    ph = 2 * np.pi * np.cumsum(f * (1 + 0.05 * np.sin(2 * np.pi * 3 * t))) / sr
    x = sum(np.sin(k * ph) / k for k in range(1, 8))
    rng = np.random.default_rng(seed)
    return (0.2 * x / np.abs(x).max() + 0.002 * rng.standard_normal(len(t))).astype(np.float32)


def test_world_variants_shift_and_flatten_pitch():
    sr = 16_000
    x = _voiced(sr)
    v = fz.world_variants(x, sr, [4, -4])
    assert set(v) == {"w0", "s+4", "s-4", "flat"} and all(len(y) == len(x) for y in v.values())
    f = {n: extract_f0_pyworld(y, sr, 320, 50, 550) for n, y in v.items()}
    assert abs(fz.pitch_shift_st(f["s+4"], f["w0"]) - 4) < 0.5
    assert abs(fz.pitch_shift_st(f["s-4"], f["w0"]) + 4) < 0.5
    assert fz.intonation_ratio(f["flat"], f["w0"]) < 0.5


def test_metrics():
    a = np.array([100.0] * 20 + [0.0] * 5)
    assert fz.pitch_shift_st(a * 2, a) == pytest.approx(12.0)
    assert np.isnan(fz.pitch_shift_st(a[:5], a[:5]))  # too few voiced frames
    assert fz.token_agreement(torch.tensor([[1, 2, 3, 4]]), torch.tensor([[1, 0, 3, 4, 9]])) == 0.75
    flat = np.array([100.0] * 20 + [60.0])  # one edge outlier does not count
    var = np.array([80.0, 120.0] * 10)
    assert fz.intonation_ratio(flat, var) == pytest.approx(0.0)


def test_aggregate_corpus_wer():
    per = {"a": {"v/c/wer_errors": 1.0, "v/c/wer_words": 10.0, "v/c/shift_st": 1.0},
           "b": {"v/c/wer_errors": 3.0, "v/c/wer_words": 10.0, "v/c/shift_st": 3.0}}
    agg = fz.aggregate(per)
    assert agg["v/c/wer"]["mean"] == pytest.approx(20.0)
    assert agg["v/c/shift_st"]["mean"] == pytest.approx(2.0)


def test_end_to_end_two_shards_and_merge(tmp_path, monkeypatch):
    import importlib.util

    import soundfile as sf

    from fact.config import tiny_config
    from fact.model import FaCT
    from fact.train import stage_d_wave_decoder as sd

    cfg = tiny_config()
    cfg.bottleneck.residual_capacity = 2.0
    cfg.bottleneck.residual_dropout = 0.5
    torch.manual_seed(0)
    fck = tmp_path / "fact.pt"
    torch.save({"model": FaCT(cfg).state_dict(), "step": 1, "config": cfg.to_dict()}, fck)
    ev = tmp_path / "ls"
    ev.mkdir()
    lines = []
    for i, (spk, f) in enumerate([(11, 120), (11, 130), (22, 200), (22, 210)]):
        p = ev / f"{i:04d}_{spk}-1-{i}.wav"
        sf.write(p, _voiced(16_000, 1.2, f, seed=i), 16_000)
        p.with_suffix(".txt").write_text("hello world")
        lines.append(f"{p}\thello world\n")
    man = tmp_path / "m.tsv"
    man.write_text("".join(lines))
    ck = tmp_path / "wd"
    monkeypatch.setattr(sys, "argv", ["x", "--tiny", "--fact-ckpt", str(fck), "--manifest", str(man),
                                      "--dev-manifest", str(man), "--ckpt-dir", str(ck),
                                      "--device", "cpu", "--steps", "2"])
    sd.main()
    dec = ck / "step_0000002.pt"

    from fact.baselines import base as registry

    class SignCodec:  # stand-in single-stream codec for the token-agreement path
        sample_rate = 8_000

        def __init__(self, device="cpu", frames=10):
            self.frames = int(frames)

        def tokenize(self, wav):
            x = wav[0, : wav.shape[-1] // self.frames * self.frames].reshape(-1, self.frames)
            return [(x.mean(1) > 0).long()[None]]
    monkeypatch.setitem(registry._REGISTRY, "signcodec", lambda device="cpu", **kw: SignCodec(device, **kw))

    spec = importlib.util.spec_from_file_location("ef", "scripts/eval_factorization.py")
    ef = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ef)
    for shard in (0, 1):
        ef.main(["--decoder", str(dec), "--eval-dir", str(ev), "--out",
                 str(tmp_path / f"r_shard{shard}.json"), "--shard", str(shard),
                 "--num-shards", "2", "--device", "cpu", "--baselines", "signcodec:frames=40"])
    res = fz.merge(sorted(str(p) for p in tmp_path.glob("r_shard*.json")), tmp_path / "r.json")
    assert res["n_utterances"] + len(res["skipped"]) == 4
    agg = res["aggregate"]
    for key in ("discrete_only/prosody+4/shift_st", "full/content-4/shift_st",
                "tokens/content_agree/s+4", "tokens/prosody_agree/flat",
                "discrete_only/prosody_flat/intonation_ratio", "full/vc/expected_st",
                "tokens/signcodec_frames40_agree/s-4"):
        assert key in agg, key
    with pytest.raises(SystemExit):  # a missing shard is refused
        fz.merge([str(tmp_path / "r_shard0.json")], tmp_path / "bad.json")
    with pytest.raises(SystemExit):  # bad arguments fail before any loading
        ef.parse_args(["--decoder", str(dec), "--eval-dir", str(ev), "--out", "x", "--shifts", "0"])
    assert json.loads((tmp_path / "r.json").read_text())["num_shards"] == 2
