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

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..config import FaCTConfig
from ..model import Batch
from .prosody_targets import prosody_targets_from_mel_and_f0


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
        if self.crop_tokens is not None:
            s = self.cfg.encoder.frame_stack
            t_mel = self.crop_tokens * s
            if mel.shape[0] > t_mel:
                start = self.rng.randrange(0, mel.shape[0] - t_mel + 1)
                mel = mel[start : start + t_mel]
                f0 = f0[start : start + t_mel]
        return {"mel": mel, "f0": f0, "text": item["text"]}


def collate(items: list[dict], cfg: FaCTConfig) -> Batch:
    from .synthetic import collate as _collate
    return _collate(items, cfg)
