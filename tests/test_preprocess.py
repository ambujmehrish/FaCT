"""End-to-end data pipeline: wavs on disk -> manifest -> shards -> stats
-> ShardedSpeech -> training batch."""

import json
import math
import wave
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fact.config import tiny_config
from fact.data.dataset import ShardedSpeech, collate
from fact.data.manifest import scan_generic, scan_libritts, write_manifest
from fact.data.preprocess import compute_stats, preprocess
from fact.data.prosody_targets import extract_f0_autocorr


def write_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    pcm = (np.clip(wav, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(pcm.tobytes())


def make_corpus(root: Path, sr: int, n: int = 5, seconds: float = 1.5) -> None:
    root.mkdir(parents=True)
    t = np.arange(int(seconds * sr)) / sr
    for i in range(n):
        f0 = 150.0 + 30.0 * i
        wav = 0.6 * np.sin(2 * math.pi * f0 * t) + 0.2 * np.sin(2 * math.pi * 2 * f0 * t)
        write_wav(root / f"utt_{i}.wav", wav, sr)
        (root / f"utt_{i}.txt").write_text(f"hello world {i}")


def test_f0_autocorr_on_sine():
    sr = 16_000
    t = np.arange(sr) / sr
    wav = 0.5 * np.sin(2 * math.pi * 220.0 * t)
    f0 = extract_f0_autocorr(wav, sr, hop_length=320, f0_min=50, f0_max=550)
    voiced = f0[f0 > 0]
    assert len(voiced) > 0.8 * len(f0)
    assert abs(np.median(voiced) - 220.0) < 10.0
    # Silence must be unvoiced.
    silent = extract_f0_autocorr(np.zeros(sr), sr, 320, 50, 550)
    assert (silent == 0).all()


def test_manifest_scan(tmp_path):
    make_corpus(tmp_path / "corpus", sr=16_000, n=3)
    items = scan_generic(tmp_path / "corpus")
    assert len(items) == 3
    assert all(text.startswith("hello world") for _, text in items)
    kept, skipped = write_manifest(items, tmp_path / "m.tsv")
    assert kept == 3 and skipped == 0


def test_manifest_libritts_layout(tmp_path):
    root = tmp_path / "LibriTTS_R" / "train-clean" / "19" / "198"
    root.mkdir(parents=True)
    sr = 16_000
    t = np.arange(sr) / sr
    write_wav(root / "19_198_000000_000000.wav",
              0.5 * np.sin(2 * math.pi * 200 * t), sr)
    (root / "19_198_000000_000000.normalized.txt").write_text("normalized text")
    (root / "19_198_000000_000000.original.txt").write_text("Original, text!")
    items = scan_libritts(tmp_path / "LibriTTS_R")
    assert items == [(str(root / "19_198_000000_000000.wav"), "normalized text")]


def test_preprocess_end_to_end(tmp_path):
    cfg = tiny_config()
    sr = cfg.audio.sample_rate
    make_corpus(tmp_path / "corpus", sr=sr, n=5)
    items = scan_generic(tmp_path / "corpus")
    write_manifest(items, tmp_path / "m.tsv")

    out = tmp_path / "shards"
    stats = preprocess(str(tmp_path / "m.tsv"), str(out), cfg,
                       shard_size=2, workers=1, min_sec=0.5, max_sec=10.0)
    shard_files = sorted(out.glob("shard_*.pt"))
    assert len(shard_files) == 3  # 5 utts, shard_size 2
    assert stats["n_utterances"] == 5
    assert stats["hours"] > 0
    assert json.loads((out / "stats.json").read_text())["mel_mean"] == stats["mel_mean"]

    # Shard contents match the dataset contract.
    item = torch.load(shard_files[0], map_location="cpu", weights_only=True)[0]
    assert item["mel"].dtype == torch.float16
    assert item["f0"].shape[0] == item["mel"].shape[0]
    assert item["text"].min() >= 1  # 0 reserved for blank

    # Resume: re-run skips all existing shards, stats unchanged.
    stats2 = preprocess(str(tmp_path / "m.tsv"), str(out), cfg,
                        shard_size=2, workers=1, min_sec=0.5, max_sec=10.0)
    assert stats2["n_utterances"] == 5

    # Shards feed straight into training.
    ds = ShardedSpeech(str(out), cfg, crop_tokens=8)
    loader = DataLoader(ds, batch_size=2, collate_fn=partial(collate, cfg=cfg))
    batch = next(iter(loader))
    assert batch.mel.shape[0] == 2
    assert batch.mel.shape[1] == 8 * cfg.encoder.frame_stack  # token-count-fixed crop
    assert batch.f0_bins.shape[1] == 8
    from fact.model import FaCT
    logs = FaCT(cfg).training_step(batch)
    assert torch.isfinite(logs["loss"])


def test_preprocess_duration_filter(tmp_path):
    cfg = tiny_config()
    sr = cfg.audio.sample_rate
    make_corpus(tmp_path / "corpus", sr=sr, n=2, seconds=0.3)  # too short
    items = scan_generic(tmp_path / "corpus")
    write_manifest(items, tmp_path / "m.tsv")
    stats = preprocess(str(tmp_path / "m.tsv"), str(tmp_path / "shards"), cfg,
                       shard_size=4, workers=1, min_sec=1.0, max_sec=30.0)
    assert stats["n_utterances"] == 0


def test_preprocess_parallel_workers(tmp_path):
    cfg = tiny_config()
    make_corpus(tmp_path / "corpus", sr=cfg.audio.sample_rate, n=4)
    write_manifest(scan_generic(tmp_path / "corpus"), tmp_path / "m.tsv")
    stats = preprocess(str(tmp_path / "m.tsv"), str(tmp_path / "shards"), cfg,
                       shard_size=2, workers=2, min_sec=0.5, max_sec=10.0)
    assert stats["n_utterances"] == 4


def test_stats_only(tmp_path):
    cfg = tiny_config()
    make_corpus(tmp_path / "corpus", sr=cfg.audio.sample_rate, n=2)
    write_manifest(scan_generic(tmp_path / "corpus"), tmp_path / "m.tsv")
    out = tmp_path / "shards"
    preprocess(str(tmp_path / "m.tsv"), str(out), cfg, shard_size=2,
               workers=1, min_sec=0.5, max_sec=10.0)
    stats = compute_stats(out, cfg.audio.hop_length, cfg.audio.sample_rate)
    assert stats["n_utterances"] == 2
    assert -20 < stats["mel_mean"] < 5
