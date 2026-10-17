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

import bisect
import json
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

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
    if stats.get("partial") or "mel_mean" not in stats:
        raise SystemExit(
            f"{p} is a partial stamp from chunked preprocessing (--skip-stats). "
            f"Finish with `python -m fact.data.preprocess --out {shard_dir} "
            f"--stats-only` before training."
        )
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


INDEX_NAME = "index.json"


def write_shard_index(shard_dir: Path, counts: dict[str, int]) -> None:
    """Atomically write {shard file name: item count}. Best effort: a
    read-only directory just means the counts get rebuilt next time."""
    tmp = shard_dir / f"{INDEX_NAME}.{os.getpid()}.tmp"
    try:
        tmp.write_text(json.dumps(counts, indent=0, sort_keys=True))
        tmp.rename(shard_dir / INDEX_NAME)
    except OSError:
        tmp.unlink(missing_ok=True)


def shard_counts(shard_dir: Path, paths: list[Path]) -> list[int]:
    """Item count per shard, from index.json when it covers every shard.

    Otherwise each shard is loaded once to count it and the result is cached
    to index.json, so only the first run over a shard directory pays that.
    """
    p = shard_dir / INDEX_NAME
    if p.exists():
        index = json.loads(p.read_text())
        if all(x.name in index for x in paths):
            return [int(index[x.name]) for x in paths]
    counts = [len(torch.load(x, map_location="cpu", weights_only=True)) for x in paths]
    write_shard_index(shard_dir, {x.name: n for x, n in zip(paths, counts)})
    return counts


class ShardedSpeech(Dataset):
    """Items are addressed by a global index; consecutive indices live in the
    same shard. Only the most recently used shard is kept in memory, so
    iterate shard by shard (ShardGroupedSampler) - uniform random access
    would reload a ~50 MB shard for almost every item."""

    def __init__(self, shard_dir: str, cfg: FaCTConfig,
                 crop_tokens: int | None = None, seed: int = 0):
        self.cfg = cfg
        self.crop_tokens = crop_tokens
        self.rng = random.Random(seed)
        root = Path(shard_dir)
        self.paths = sorted(root.glob("*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"No .pt shards in {shard_dir}")
        self.counts = shard_counts(root, self.paths)
        self.starts = [0]
        for n in self.counts:
            self.starts.append(self.starts[-1] + n)
        self._cache_si: int | None = None
        self._cache: list[dict] | None = None

    def __len__(self) -> int:
        return self.starts[-1]

    def shard_range(self, si: int) -> range:
        return range(self.starts[si], self.starts[si + 1])

    def _load(self, si: int) -> list[dict]:
        if si != self._cache_si:
            self._cache = torch.load(self.paths[si], map_location="cpu", weights_only=True)
            self._cache_si = si
        return self._cache

    def __getitem__(self, i: int) -> dict:
        si = bisect.bisect_right(self.starts, i) - 1
        item = self._load(si)[i - self.starts[si]]
        from .preprocess import dequantize_mel
        mel = dequantize_mel(item)  # fp16 or int8 storage, transparently
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


class ShardGroupedSampler(Sampler[int]):
    """Shuffled order that visits one shard at a time.

    Each epoch shuffles the shard order (the same on every rank), gives each
    DDP rank every world_size-th shard, and shuffles items within a shard.
    Shards are written from a shuffled manifest, so every shard is already a
    random sample of the corpus and batching within a shard loses no
    randomness. Ranks may get slightly different epoch lengths; the trainer
    counts steps, not epochs, and restarts a rank's loader when it runs out.
    """

    def __init__(self, dataset: ShardedSpeech, shuffle: bool = True, seed: int = 0,
                 rank: int = 0, world_size: int = 1):
        if len(dataset.counts) < world_size:
            raise ValueError(
                f"{len(dataset.counts)} shards cannot be split across {world_size} "
                f"ranks; use fewer GPUs or smaller --shard-size when preprocessing")
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        self.skip = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_skip(self, n: int) -> None:
        """Drop the first n indices of the NEXT iteration only (resume mid-epoch).

        Skipping happens at the index level, so no skipped sample is loaded.
        """
        self.skip = max(0, int(n))

    def _my_shards(self) -> list[int]:
        order = list(range(len(self.dataset.counts)))
        if self.shuffle:
            random.Random(self.seed * 1_000_003 + self.epoch).shuffle(order)
        return order[self.rank :: self.world_size]

    def __iter__(self):
        skip, self.skip = self.skip, 0
        for si in self._my_shards():
            n = self.dataset.counts[si]
            if skip >= n:  # whole shard already consumed: never load it
                skip -= n
                continue
            items = list(self.dataset.shard_range(si))
            if self.shuffle:
                random.Random((self.seed * 1_000_003 + self.epoch) * 7_919 + si).shuffle(items)
            yield from items[skip:]
            skip = 0

    def __len__(self) -> int:
        return sum(self.dataset.counts[si] for si in self._my_shards())


def collate(items: list[dict], cfg: FaCTConfig) -> Batch:
    from .synthetic import collate as _collate
    return _collate(items, cfg)
