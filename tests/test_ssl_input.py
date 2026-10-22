"""#4 SSL-input pipeline: audio index, aligned audio loading, frame
alignment of the SSL features, and the encoder's SSL input."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fact.config import AudioConfig, FaCTConfig, tiny_config
from fact.model import FaCT
from fact.modules.ssl import SSLFeatures, ssl_samples


def _bai():
    spec = importlib.util.spec_from_file_location("bai", Path("scripts/build_audio_index.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_build_audio_index(tmp_path):
    shards = tmp_path / "sh"
    shards.mkdir()
    man = []
    for c in ("c000", "c001"):
        lines = [f"/data/{c}/u{k}.mp3\ttext ): odd {k}\n" for k in range(6)]
        (tmp_path / f"train_{c}.tsv").write_text("".join(lines))
        man.append(tmp_path / f"train_{c}.tsv")
    # shard size 4: c000 -> shards 0 (u0-3), 1 (u4-5); u1 filtered, u4 error.
    (shards / "drops.log").write_text(
        "shard 0: filtered(0.4s): /data/c000/u1.mp3\n"
        "shard 1: error(RuntimeError: Cannot decode /x: bad): /data/c000/u4.mp3\ttext ): odd 4\n")
    counts = {"shard_c000_000000.pt": 3, "shard_c000_000001.pt": 1,
              "shard_c001_000000.pt": 4, "shard_c001_000001.pt": 2}
    (shards / "index.json").write_text(json.dumps(counts))
    rep = _bai().build(shards, man, True, 4, shards / "audio_index")
    assert rep["items"] == 10 and rep["shards"] == 4
    got = (shards / "audio_index" / "shard_c000_000000.txt").read_text().split()
    assert got == ["/data/c000/u0.mp3", "/data/c000/u2.mp3", "/data/c000/u3.mp3"]
    assert (shards / "audio_index" / "shard_c000_000001.txt").read_text().split() == ["/data/c000/u5.mp3"]
    counts["shard_c001_000001.pt"] = 1  # inconsistent with the drops -> abort
    (shards / "index.json").write_text(json.dumps(counts))
    with pytest.raises(SystemExit):
        _bai().build(shards, man, True, 4, shards / "audio_index")


def _shards_with_audio(tmp_path, n=3):
    import soundfile as sf
    from fact.modules.mel import LogMelSpectrogram
    cfg = tiny_config()
    cfg.audio = AudioConfig()
    mel_fn = LogMelSpectrogram(cfg.audio)
    rng = np.random.default_rng(0)
    items, paths = [], []
    for k in range(n):
        x = (0.1 * rng.standard_normal(24_000 * (1 + k))).astype("float32")
        p = tmp_path / f"a{k}.wav"
        sf.write(p, x, 24_000)
        mel = mel_fn(torch.from_numpy(x)[None])[0]
        items.append({"mel": mel.half(), "f0": torch.zeros(mel.shape[0]),
                      "text": torch.tensor([5, 6, 7])})
        paths.append(str(p))
    sh = tmp_path / "sh"
    sh.mkdir()
    torch.save(items, sh / "shard_000000.pt")
    (sh / "index.json").write_text(json.dumps({"shard_000000.pt": n}))
    (sh / "audio_index").mkdir()
    (sh / "audio_index" / "shard_000000.txt").write_text("\n".join(paths) + "\n")
    return cfg, sh, paths


def test_dataset_audio_alignment_and_collate(tmp_path):
    from fact.data.dataset import ShardedSpeech, collate
    cfg, sh, paths = _shards_with_audio(tmp_path)
    ds = ShardedSpeech(str(sh), cfg, audio_index=str(sh / "audio_index"))
    items = [ds[i] for i in range(3)]
    b = collate(items, cfg)
    lens = b.mel_lengths.tolist()
    assert b.wav_lengths.tolist() == [ssl_samples(L) for L in lens]
    assert b.wav16.shape == (3, ssl_samples(max(lens)))
    # A misaligned index (paths swapped) is caught per item.
    (sh / "audio_index" / "shard_000000.txt").write_text("\n".join(reversed(paths)) + "\n")
    ds = ShardedSpeech(str(sh), cfg, audio_index=str(sh / "audio_index"))
    with pytest.raises(ValueError, match="misaligned"):
        ds[0]


def _tiny_wavlm():
    from transformers import WavLMConfig, WavLMModel
    c = WavLMConfig(hidden_size=32, num_hidden_layers=3, num_attention_heads=2,
                    intermediate_size=64, conv_dim=(8,) * 7, feat_extract_norm="layer",
                    do_stable_layer_norm=True)
    torch.manual_seed(0)
    return WavLMModel(c)


def test_ssl_frames_align_and_batching_is_exact():
    f = SSLFeatures(layer=2, model=_tiny_wavlm())
    for L in (4, 17, 40):
        assert f(torch.randn(1, ssl_samples(L))).shape == (1, L, 32)
    a, b = torch.randn(ssl_samples(17)), torch.randn(ssl_samples(40))
    batch = torch.zeros(2, b.shape[0])
    batch[0, : a.shape[0]] = a
    batch[1] = b
    out = f(batch, torch.tensor([a.shape[0], b.shape[0]]))
    assert torch.allclose(out[0, :17], f(a[None])[0], atol=1e-4)
    assert len(f.model.encoder.layers) == 2 and not any(p.requires_grad for p in f.parameters())


def test_encoder_ssl_input_required_and_used():
    from fact.data.synthetic import SyntheticSpeech, collate
    cfg = tiny_config()
    cfg.encoder.ssl_model, cfg.encoder.ssl_dim = "tiny", 32
    model = FaCT(cfg)
    ds = SyntheticSpeech(cfg, n_items=2, seconds=1.0)
    b = collate([ds[0], ds[1]], cfg)
    with pytest.raises(ValueError, match="expects"):
        model.training_step(b)
    b.ssl = torch.randn(2, b.mel.shape[1], 32)
    logs = model.training_step(b)
    assert torch.isfinite(logs["loss"])
    logs["loss"].backward()
    assert model.encoder.ssl_proj.weight.grad is not None
    # mel-only model refuses ssl input.
    plain = FaCT(tiny_config())
    with pytest.raises(ValueError):
        plain.encode_mel(b.mel, b.ssl)


def test_ssl_config_differs_from_noncausal_only_in_ssl():
    a = FaCTConfig.from_yaml("configs/ablation_noncausal.yaml").to_dict()
    b = FaCTConfig.from_yaml("configs/fact_ssl_noncausal.yaml").to_dict()
    diff = {(s, k) for s in a for k in a[s] if a[s][k] != b[s][k]}
    assert diff == {("encoder", "ssl_model")}


def test_trainer_runs_with_ssl_features(tmp_path, monkeypatch):
    from functools import partial

    from torch.utils.data import DataLoader

    import fact.modules.ssl as ssl_mod
    from fact.data.dataset import ShardedSpeech, collate
    from fact.train.trainer import Trainer
    cfg, sh, _ = _shards_with_audio(tmp_path)
    cfg.encoder.ssl_model, cfg.encoder.ssl_dim = "tiny", 32
    cfg.train.max_steps, cfg.train.log_every = 2, 1
    real = SSLFeatures
    monkeypatch.setattr(ssl_mod, "SSLFeatures", lambda model_id, layer: real(layer=2, model=_tiny_wavlm()))
    ds = ShardedSpeech(str(sh), cfg, audio_index=str(sh / "audio_index"))
    loader = DataLoader(ds, batch_size=2, collate_fn=partial(collate, cfg=cfg), drop_last=True)
    tr = Trainer(FaCT(cfg), cfg, device="cpu")
    assert tr.ssl is not None
    logs = tr.fit(loader, max_steps=2)
    assert tr.state.step == 2 and np.isfinite(logs["loss"])


def test_eval_fact_recon_feeds_ssl(tmp_path, monkeypatch):
    import sys

    import soundfile as sf

    import fact.modules.ssl as ssl_mod
    cfg = tiny_config()
    cfg.audio = AudioConfig()
    cfg.encoder.ssl_model, cfg.encoder.ssl_dim = "tiny", 32
    cfg.bottleneck.residual_capacity = 2.0
    ck = tmp_path / "ck.pt"
    torch.save({"model": FaCT(cfg).state_dict(), "step": 1, "config": cfg.to_dict()}, ck)
    d = tmp_path / "w"
    d.mkdir()
    sf.write(d / "u0.wav", (0.1 * np.random.default_rng(0).standard_normal(16_000)).astype("float32"), 16_000)
    real = SSLFeatures
    monkeypatch.setattr(ssl_mod, "ssl_for_config", lambda c, dev: real(layer=2, model=_tiny_wavlm()))
    out = tmp_path / "r.json"
    monkeypatch.setattr(sys, "argv", ["x", "--ckpt", str(ck), "--eval-dir", str(d), "--out", str(out),
                                      "--nfe", "2", "--device", "cpu", "--views", "full,discrete_only"])
    spec = importlib.util.spec_from_file_location("e", Path("scripts/eval_fact_recon.py"))
    e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e)
    e.main()
    assert json.loads(out.read_text())["views"]["full"]["mel_l1"]["n"] == 1
