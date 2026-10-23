"""Speaker-reference fix: disjoint training reference, other-utterance
evaluation reference, and tables restricted to shared utterances."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from fact.config import tiny_config
from fact.data.synthetic import SyntheticSpeech, collate
from fact.model import FaCT


def test_disjoint_reference_never_overlaps_the_loss():
    mel = torch.randn(4, 40, 8)
    lengths = torch.tensor([40, 32, 20, 8])
    torch.manual_seed(0)
    ref, r, in_ref = FaCT._disjoint_reference(mel, lengths)
    for i in range(4):
        L = int(lengths[i])
        assert 0.25 * L - 1 <= int(r[i]) <= 0.5 * L
        idx = in_ref[i].nonzero().flatten()
        assert idx.min() == 0 or idx.max() == L - 1          # a prefix or a suffix
        assert torch.equal(ref[i, : r[i]], mel[i, idx])       # the reference IS those frames
        assert in_ref[i, L:].sum() == 0


def test_disjoint_mode_trains_and_differs_from_same():
    cfg = tiny_config()
    ds = SyntheticSpeech(cfg, n_items=2, seconds=1.0)
    b = collate([ds[0], ds[1]], cfg)
    torch.manual_seed(0)
    m = FaCT(cfg)
    torch.manual_seed(1)
    same = m.training_step(b)["recon"]
    cfg.speaker.ref_mode = "disjoint"
    torch.manual_seed(1)
    dis = m.training_step(b)["recon"]
    assert torch.isfinite(dis) and dis != same
    cfg.speaker.ref_mode = "bogus"
    with pytest.raises(ValueError):
        m.training_step(b)


def test_wrapper_other_reference_uses_partner(tmp_path, monkeypatch):
    import fact.baselines.fact_vocoded as fv
    ref = fv._SpeakerRef()
    ref._init_ref("other", 0.0)
    from fact.config import AudioConfig
    from fact.modules.mel import LogMelSpectrogram
    ref.mel_fn, ref.sample_rate, ref.device = LogMelSpectrogram(AudioConfig()), 24_000, "cpu"
    w1, w2 = torch.randn(24_000), torch.randn(48_000)
    ref.set_eval_set([("0000_10-1-0.wav", w1, 24_000), ("0001_10-1-1.wav", w2, 24_000),
                      ("0002_20-1-0.wav", w1, 24_000)])
    mel = torch.zeros(1, 5, 100)
    got = ref._ref_mel(mel, "0000_10-1-0.wav")
    assert got.shape[1] == 48_000 // 480 + 1           # the partner's mel, not the input's
    with pytest.raises(KeyError, match="partner"):
        ref._ref_mel(mel, "0002_20-1-0.wav")            # singleton speaker
    with pytest.raises(ValueError):
        fv._SpeakerRef()._init_ref("prefix", 0.0)


def test_recon_table_restricts_to_common_utterances(tmp_path):
    spec = importlib.util.spec_from_file_location("mrt", Path("scripts/make_recon_table.py"))
    mrt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mrt)
    for name, utts in (("a", ["u1", "u2", "u3"]), ("b", ["u2", "u3"])):
        json.dump({"baseline": name, "metadata": {"discrete_bps": 500},
                   "per_utterance": {u: {"extras": {"mel_l1_band": 1.0}} for u in utts}},
                  open(tmp_path / f"{name}.json", "w"))
    import sys
    out = tmp_path / "t.md"
    sys.argv = ["x", "--title", "t", "--baselines", str(tmp_path / "a.json"), str(tmp_path / "b.json"),
                "--out", str(out)]
    mrt.main()
    text = out.read_text()
    assert "n = 2" in text and "restricted" in text
