"""Sharded on-disk dataset for real training runs.

Expects preprocessed shards produced by `fact/data/preprocess.py`: each
shard is a .pt file holding a list of dicts with keys
  mel:  (T_mel, n_mels) float16/float32 log-mel
  f0:   (T_mel,) float32 F0 in Hz (0 = unvoiced)
  text: (S,) int64 label ids (byte-level by default; 0 reserved for blank)
Cropping to a fixed token count (not seconds) is a first-class option -
this is the RQ4 token-count-fixed training knob.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..config import FaCTConfig
from ..model import Batch
from .prosody_targets import prosody_targets_from_mel_and_f0


def verify_mel_stats(shard_dir: str, cfg: FaCTConfig,
                     tol_mean: float = 0.5, tol_std: float = 0.5,
                     allow_mismatch: bool = False) -> None:
    """Refuse to train when config normalization disagrees with the corpus.

    Preprocessing writes stats.json (true mel mean/std). If the config's
    audio.mel_mean/mel_std are off, flow-matching targets are mis-scaled for
    the entire run - a silent, expensive failure. Hard error unless
    explicitly overridden.
    """
    p = Path(shard_dir) / "stats.json"
    if not p.exists():
        raise SystemExit(
            f"{p} not found. Run preprocessing to completion (it writes "
            f"stats.json), or `python -m fact.data.preprocess --out "
            f"{shard_dir} --stats-only` to regenerate it."
        )
    stats = json.loads(p.read_text())
    if allow_mismatch:
        return
    d_mean = abs(stats["mel_mean"] - cfg.audio.mel_mean)
    d_std = abs(stats["mel_std"] - cfg.audio.mel_std)
    if d_mean > tol_mean or d_std > tol_std:
        raise SystemExit(
            f"Config mel normalization (mean={cfg.audio.mel_mean}, "
            f"std={cfg.audio.mel_std}) does not match the corpus stats in "
            f"{p} (mean={stats['mel_mean']}, std={stats['mel_std']}). "
            f"Set audio.mel_mean/mel_std in your config to the corpus "
            f"values, or pass --allow-stats-mismatch to proceed anyway."
        )
    # Preprocessing fingerprint: shards built under an older/different config
    # (sample rate, mel resolution, text vocab) are silently incompatible -
    # refuse rather than mix encodings.
    expected = {
        "sample_rate": cfg.audio.sample_rate,
        "hop_length": cfg.audio.hop_length,
        "n_mels": cfg.audio.n_mels,
        "text_vocab_size": cfg.ctc.vocab_size,
    }
    fp = stats.get("fingerprint")
    if fp != expected:
        raise SystemExit(
            f"Shard fingerprint mismatch in {p}: shards were preprocessed "
            f"with {fp}, config expects {expected}. Re-run preprocessing "
            f"into a fresh directory (old shards keep old encodings even "
            f"after a config change), or pass --allow-stats-mismatch if you "
            f"are certain this is benign."
        )


class ShardedSpeech(Dataset):
    def __init__(self, shard_dir: str, cfg: FaCTConfig,
                 crop_tokens: int | None = None, seed: int = 0):
        self.cfg = cfg
        self.crop_tokens = crop_tokens
        self.rng = random.Random(seed)
        self.paths = sorted(Path(shard_dir).glob("*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"No .pt shards in {shard_dir}")
        self._index: list[tuple[int, int]] = []
        for si, p in enumerate(self.paths):
            n = len(torch.load(p, map_location="cpu", weights_only=True))
            self._index.extend((si, i) for i in range(n))
        self._cache_si: int | None = None
        self._cache: list[dict] | None = None

    def __len__(self) -> int:
        return len(self._index)

    def _load(self, si: int) -> list[dict]:
        if si != self._cache_si:
            self._cache = torch.load(self.paths[si], map_location="cpu", weights_only=True)
            self._cache_si = si
        return self._cache

    def __getitem__(self, i: int) -> dict:
        si, ii = self._index[i]
        item = self._load(si)[ii]
        mel = item["mel"].float()
        f0 = item["f0"].float()
        text = item["text"]
        if self.crop_tokens is not None:
            s = self.cfg.encoder.frame_stack
            t_mel = self.crop_tokens * s
            if mel.shape[0] > t_mel:
                start = self.rng.randrange(0, mel.shape[0] - t_mel + 1)
                mel = mel[start : start + t_mel]
                f0 = f0[start : start + t_mel]
                # The transcript no longer matches the cropped audio; drop it
                # (text_length 0 excludes the item from CTC/p2t losses)
                # rather than training CTC against absent speech.
                text = torch.zeros(0, dtype=torch.long)
        return {"mel": mel, "f0": f0, "text": text}


def collate(items: list[dict], cfg: FaCTConfig) -> Batch:
    from .synthetic import collate as _collate
    return _collate(items, cfg)
