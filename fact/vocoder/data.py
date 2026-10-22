"""Random waveform segments for vocoder training, streamed from a manifest.

The Emilia train manifest is several GB, so it is never loaded into memory:
each draw seeks to a uniformly random byte offset and takes the next full
line. That weights a line by the byte length of the line before it - a
transcript length, unrelated to the audio a vocoder learns from - and
never draws line 1 (one of millions); stated here because it is not
exactly uniform.

Every (rank, worker) has its own RNG stream, seeded from the config seed
and the resume step, so a resumed run draws fresh segments instead of
replaying the first ones. Sampling is with replacement, so there is no
epoch to resume a position in.

Unreadable files are skipped but counted; if more than 1% of the first
1,000 draws fail, the stream raises (a path or codec problem, not a few
bad files).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..data.preprocess import load_audio


class RandomSegments(IterableDataset):
    def __init__(self, manifest: str, sample_rate: int, segment: int, seed: int,
                 rank: int = 0, step_offset: int = 0, gain_db: tuple = (-6.0, -1.0),
                 max_fail_rate: float = 0.01):
        self.manifest = manifest
        self.sample_rate = sample_rate
        self.segment = segment
        self.seed = seed
        self.rank = rank
        self.step_offset = step_offset
        self.gain_db = gain_db
        self.max_fail_rate = max_fail_rate
        self.size = os.path.getsize(manifest)
        if self.size == 0:
            raise ValueError(f"empty manifest {manifest}")

    def _rng(self) -> random.Random:
        info = get_worker_info()
        wid = info.id if info else 0
        return random.Random(f"{self.seed}/{self.rank}/{wid}/{self.step_offset}")

    def _draw_path(self, fh, rng: random.Random) -> str:
        while True:
            fh.seek(rng.randrange(self.size))
            fh.readline()  # finish the partial line
            line = fh.readline()
            if line.strip():
                return line.split(b"\t", 1)[0].decode("utf-8").strip()

    def __iter__(self):
        rng = self._rng()
        tries = fails = 0
        with open(self.manifest, "rb") as fh:
            while True:
                path = self._draw_path(fh, rng)
                tries += 1
                try:
                    wav = load_audio(path, self.sample_rate)
                except RuntimeError as e:
                    fails += 1
                    if tries >= 1000 and fails > self.max_fail_rate * tries:
                        raise RuntimeError(f"{fails}/{tries} manifest audio files unreadable "
                                           f"(last: {e})") from e
                    continue
                if len(wav) < self.segment:
                    continue  # shorter than one training segment
                peak = float(np.abs(wav).max())
                if peak < 1e-4:
                    continue  # digital silence: no signal to learn from
                gain = 10 ** (rng.uniform(*self.gain_db) / 20)
                start = rng.randrange(len(wav) - self.segment + 1)
                seg = wav[start:start + self.segment] * (gain / peak)
                yield torch.from_numpy(np.ascontiguousarray(seg, dtype=np.float32))


def load_dev_utterances(manifest: str, n: int, sample_rate: int,
                        max_seconds: float = 10.0) -> list[tuple[str, torch.Tensor]]:
    """First n readable dev utterances (deterministic), cut to max_seconds."""
    out = []
    with open(manifest, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            path = line.split("\t", 1)[0].strip()
            wav = load_audio(path, sample_rate)[: int(max_seconds * sample_rate)]
            out.append((path, torch.from_numpy(wav)))
            if len(out) >= n:
                break
    if len(out) < n:
        raise ValueError(f"{manifest} has only {len(out)} utterances (< {n})")
    return out
