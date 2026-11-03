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


def test_reference_partners_refuses_duplicate_keys():
    import pytest

    from fact.eval.eval_set import reference_partners
    keys = ["0000_11-1-0000.wav", "0001_11-1-0001.wav"]
    assert reference_partners(keys) == {keys[0]: keys[1], keys[1]: keys[0]}
    with pytest.raises(ValueError, match="duplicate"):
        reference_partners(keys + [keys[0]])


def test_cross_speaker_partners_always_another_speaker():
    from fact.eval.eval_set import cross_speaker_partners
    keys = ["0000_10-1-0.wav", "0001_10-1-1.wav", "0002_20-2-0.wav", "0003_30-3-0.wav",
            "0004_30-3-1.wav", "0005_30-3-2.wav"]
    p = cross_speaker_partners(keys)
    assert set(p) == set(keys)
    assert all(speaker_key(k) != speaker_key(v) for k, v in p.items())
    assert p == cross_speaker_partners(list(reversed(keys)))  # order-independent
    with pytest.raises(ValueError):
        cross_speaker_partners(["0000_10-1-0.wav", "0001_10-1-1.wav"])


def test_spk_ref_xspk_mode_uses_cross_speaker_reference():
    from fact.baselines.fact_vocoded import _SpeakerRef
    r = _SpeakerRef()
    r._init_ref("xspk", 0.0)
    wavs = [("0000_10-1-0.wav", None, 16000), ("0001_20-1-0.wav", None, 16000)]
    r.set_eval_set(wavs)
    assert r.partners == {"0000_10-1-0.wav": "0001_20-1-0.wav", "0001_20-1-0.wav": "0000_10-1-0.wav"}


def test_spk_ref_mean_mode_uses_one_fixed_unit_vector():
    from fact.baselines.fact_vocoded import FixedSpeaker, _SpeakerRef
    from fact.modules.mel import LogMelSpectrogram
    cfg = tiny_config()
    torch.manual_seed(0)
    model = FaCT(cfg).eval()

    class R(_SpeakerRef):
        def _fact_model(self):
            return model
    r = R()
    r._init_ref("mean", 0.0)
    r.mel_fn, r.sample_rate, r.device = LogMelSpectrogram(cfg.audio), cfg.audio.sample_rate, "cpu"
    sr = cfg.audio.sample_rate
    wavs = [(f"000{i}_{10 + i}-1-0.wav", torch.randn(sr // 2), sr) for i in range(3)]
    mel = r.mel_fn(wavs[0][1].unsqueeze(0))
    with pytest.raises(RuntimeError):
        r._ref_mel(mel, wavs[0][0])  # before set_eval_set
    r.set_eval_set(wavs)
    assert isinstance(model.speaker_encoder, FixedSpeaker) and r.n_mean_utts == 3
    a = model.speaker_encoder(model._norm_mel(r._ref_mel(mel, wavs[0][0])))
    b = model.speaker_encoder(torch.randn(2, 7, cfg.audio.n_mels))
    assert torch.equal(a[0], b[1]) and abs(float(a[0].norm()) - 1.0) < 1e-5
    with pytest.raises(ValueError):
        r._init_ref("nobody", 0.0)
