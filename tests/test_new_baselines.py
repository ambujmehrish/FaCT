"""FACodec and ZipCodec wrappers + the FACodec head-to-head eval.

The real-checkpoint test runs only with FACT_TEST_REAL_CODECS=1 and the
codec sources/weights present (FACT_CODEC_SRC / FACT_CACHE, HF cache)."""

import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

from fact.baselines import base
from fact.eval import factorization as fz
from tests.test_factorization import _voiced


def test_registered_and_view_checked_before_loading():
    base._autoload()
    assert {"facodec", "zipcodec", "focalcodec"} <= set(base.available())
    with pytest.raises(ValueError, match="view"):
        base.build("facodec", device="cpu", view="tokens")  # fails before any import/download


def test_pinned_sources():
    from fact.baselines.lowrate import REPOS
    assert REPOS["zipcodec"][1] and REPOS["Amphion"][1]


class FakeFACodec:
    """FACodecBaseline's interface on a toy signal model: prosody code = F0
    bin of each frame, content code = energy bin; decode renders a harmonic
    tone at the prosody codes' F0 (the timbre vector scales the level)."""
    name, frame_rate_hz, discrete_bps = "facodec", 80.0, 4800.0
    hop, sr = 200, 16_000

    def stream_slices(self):
        return {"prosody": slice(0, 1), "content": slice(1, 3), "residual": slice(3, None)}

    def encode(self, wav):
        from fact.data.prosody_targets import extract_f0_pyworld
        w = wav[0].float().numpy()
        f0 = extract_f0_pyworld(w, self.sr, self.hop, 50, 550)
        T = len(w) // self.hop
        f0 = np.resize(f0, T)
        p = torch.from_numpy(np.where(f0 > 0, np.round(12 * np.log2(np.maximum(f0, 1) / 50)), 0)).long()
        e = torch.from_numpy(np.array([np.abs(w[i * self.hop:(i + 1) * self.hop]).mean() for i in range(T)]))
        c = (e * 100).long().clamp(0, 1023)
        ids = torch.stack([p, c, c, c, c, c])[:, None]  # (6, 1, T)
        return {"ids": ids, "spk": torch.tensor([[1.0 + float(e.mean())]])}

    def decode(self, ids, spk, use_residual=True):
        p = ids[0, 0].float()
        f0 = torch.where(p > 0, 50 * 2 ** (p / 12), torch.zeros_like(p))
        f = f0.repeat_interleave(self.hop)
        ph = 2 * np.pi * torch.cumsum(f, 0) / self.sr
        y = sum(torch.sin(k * ph) / k for k in range(1, 6)) * 0.1 * float(spk[0, 0])
        return y[None]


def test_eval_facodec_end_to_end_with_stand_in(tmp_path, monkeypatch):
    import soundfile as sf
    ev = tmp_path / "ls"
    ev.mkdir()
    for i, (spk, f) in enumerate([(11, 110), (11, 118), (22, 205), (22, 215)]):
        p = ev / f"{i:04d}_{spk}-1-{i}.wav"
        sf.write(p, _voiced(16_000, 1.2, f, seed=i), 16_000)
        p.with_suffix(".txt").write_text("x")

    class FakeSim:
        def __init__(self, device="cpu"):
            pass

        def __call__(self, a, b, sr):
            n = min(len(a), len(b))
            return float(torch.nn.functional.cosine_similarity(a[:n].float(), b[:n].float(), dim=0))
    import fact.eval.quality as q
    monkeypatch.setattr(q, "SpeakerSim", FakeSim)
    spec = importlib.util.spec_from_file_location("efa", "scripts/eval_facodec.py")
    efa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(efa)
    for shard in (0, 1):
        efa.main(["--eval-dir", str(ev), "--out", str(tmp_path / f"f_shard{shard}.json"),
                  "--shard", str(shard), "--num-shards", "2", "--device", "cpu"], codec=FakeFACodec())
    res = fz.merge(sorted(str(p) for p in tmp_path.glob("f_shard*.json")), tmp_path / "f.json")
    agg = res["aggregate"]
    for key in ("full/prosody+4/shift_st", "discrete_only/content-4/shift_st", "tokens/content_agree/s+4",
                "full/prosody_flat/intonation_ratio", "full/vc_cross/sim_tgt",
                "discrete_only/vcpm_cross/pitch_gap_abs_st", "pair/cross/expected_abs_st"):
        assert key in agg, key
    # the stand-in's pitch lives in its prosody codes: swapping them moves the pitch, content does not
    assert agg["full/prosody+4/shift_st"]["mean"] > 3 and abs(agg["full/content+4/shift_st"]["mean"]) < 0.5
    assert res["view_meaning"]["discrete_only"] == "no acoustic-detail codes"
    with pytest.raises(SystemExit):
        efa.parse_args(["--eval-dir", str(ev), "--out", "x", "--shifts", "0"])


@pytest.mark.skipif(os.environ.get("FACT_TEST_REAL_CODECS") != "1",
                    reason="needs FACodec/ZipCodec sources + weights (FACT_TEST_REAL_CODECS=1)")
def test_real_checkpoints_roundtrip():
    x = torch.from_numpy(_voiced(16_000, 2.0, 140))[None]
    fa = base.build("facodec", device="cpu")
    assert fa.frame_rate_hz == 80.0 and fa.discrete_bps == 4800.0
    assert base.build("facodec", device="cpu", view="no_residual").discrete_bps == 2400.0
    assert fa.reconstruct(x).shape == x.shape
    z = base.build("zipcodec", device="cpu")
    assert z.frame_rate_hz == 6.25 and z.discrete_bps == 800.0
    assert z.reconstruct(x).shape == x.shape
