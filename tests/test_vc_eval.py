"""Voice-conversion eval: target choice, WORLD shift, intonation metric, and
the end-to-end script + shard merge on a tiny model."""

import importlib.util
import sys

import numpy as np
import pytest
import torch

from fact.data.prosody_targets import extract_f0_pyworld
from fact.eval import factorization as fz
from fact.eval import vc
from tests.test_factorization import _voiced


def test_choose_targets_same_and_cross_group():
    groups = {"a": "low", "b": "low", "c": "high", "d": "high", "e": "high"}
    t = vc.choose_targets(groups)
    for s, d in t.items():
        assert d["same"] != s and groups[d["same"]] == groups[s]
        assert groups[d["cross"]] != groups[s]
    assert t == vc.choose_targets(dict(reversed(list(groups.items()))))  # order-independent
    t1 = vc.choose_targets({"a": "low", "c": "high"})
    assert t1["a"] == {"same": None, "cross": "c"}
    assert vc.pitch_groups({"a": 120.0, "b": 210.0, "c": float("nan")}, 160.0) == {"a": "low", "b": "high"}


def test_expected_shift_and_world_shift():
    assert vc.expected_shift_st(200.0, 100.0, 24.0) == pytest.approx(12.0)
    assert vc.expected_shift_st(400.0, 100.0, 12.0) == 12.0  # clipped
    sr = 16_000
    x = _voiced(sr)
    y = vc.world_shift(x, sr, 5.0)
    assert len(y) == len(x)
    f0x, f0y = (extract_f0_pyworld(z, sr, 320, 50, 550) for z in (x, y))
    assert abs(fz.pitch_shift_st(f0y, f0x) - 5.0) < 0.6
    with pytest.raises(ValueError):
        vc.world_shift(np.zeros(sr, dtype=np.float32), sr, 3.0)


def test_contour_corr():
    a = np.linspace(100.0, 200.0, 20)
    assert vc.contour_corr(a * 1.5, a) == pytest.approx(1.0)  # same intonation, other register
    assert vc.contour_corr(a[::-1].copy(), a) < -0.9  # inverted intonation
    assert np.isnan(vc.contour_corr(a[:5], a[:5]))
    assert vc.speaker_median_f0([np.array([0.0, 100, 110] * 5)]) == pytest.approx(105.0)


def test_eval_vc_end_to_end_and_merge(tmp_path, monkeypatch):
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
    spk_f = [(11, 110), (11, 115), (22, 125), (22, 120), (33, 210), (33, 220), (44, 200), (44, 205)]
    for i, (spk, f) in enumerate(spk_f):
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

    class FakeSim:  # the real one loads WavLM; cosine of mean-pooled frames is enough here
        def __init__(self, device="cpu"):
            pass

        def __call__(self, ref, hyp, sr):
            a, b = ref.float(), hyp.float()
            n = min(len(a), len(b))
            return float(torch.nn.functional.cosine_similarity(a[:n], b[:n], dim=0))
    import fact.eval.quality as q
    monkeypatch.setattr(q, "SpeakerSim", FakeSim)

    spec = importlib.util.spec_from_file_location("evc", "scripts/eval_vc.py")
    evc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evc)
    for shard in (0, 1):
        evc.main(["--decoder", str(dec), "--eval-dir", str(ev), "--out",
                  str(tmp_path / f"v_shard{shard}.json"), "--shard", str(shard),
                  "--num-shards", "2", "--device", "cpu"])
    res = fz.merge(sorted(str(p) for p in tmp_path.glob("v_shard*.json")), tmp_path / "v.json")
    assert res["n_utterances"] + len(res["skipped"]) == 8
    assert set(res["speaker_group"].values()) == {"low", "high"}
    agg = res["aggregate"]
    for key in ("full/recon/sim_src", "discrete_only/vc_cross/sim_tgt", "full/vcpm_cross/pitch_gap_st",
                "discrete_only/vcpm_same/contour_corr", "full/vc_same/rtf", "full/vc_cross/tgt_wins",
                "pair/cross/expected_st", "real/cross/sim_same_spk", "real/same/sim_cross"):
        assert key in agg, key
    # cross-group pairs need a large shift, same-group pairs a small one
    assert agg["pair/cross/expected_abs_st"]["mean"] > 6 > agg["pair/same/expected_abs_st"]["mean"]
    assert "full/vcpm_cross/pitch_gap_abs_st" in agg
    with pytest.raises(SystemExit):  # bad arguments fail before any loading
        evc.parse_args(["--decoder", str(dec), "--eval-dir", str(ev), "--out", "x", "--views", "mel"])
