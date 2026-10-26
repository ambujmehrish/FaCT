"""Token-shard loader for the TTS LMs.

Reads <codec dir>/shard_*.pt (fact.tts.tokenize) and keeps an item only if
EVERY directory in `require` also has it (same shard, same index, not None),
so LMs trained on different codecs see exactly the same utterances. The
first `n_val_shards` shard names (sorted) are held out for validation.
Batches are built by token budget (text + audio positions) within a shard
group, each DDP rank / loader worker reading a disjoint set of shards.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .model import EOT, TEXT_PAD, TTSConfig, bos, delayed_targets, pad_id


def common_shards(primary: Path, require: list[Path]) -> list[str]:
    names = {p.name for p in primary.glob("shard_*.pt")}
    for r in require:
        names &= {p.name for p in r.glob("shard_*.pt")}
    return sorted(names)


def split_shards(names: list[str], n_val: int) -> tuple[list[str], list[str]]:
    return names[n_val:], names[:n_val]


def load_items(name: str, primary: Path, require: list[Path], cfg: TTSConfig) -> list[dict]:
    rows = torch.load(primary / name, map_location="cpu", weights_only=True)
    keep = [r is not None for r in rows]
    for r in require:
        other = torch.load(r / name, map_location="cpu", weights_only=True)
        if len(other) != len(rows):
            raise ValueError(f"{name}: {len(rows)} items in {primary}, {len(other)} in {r}")
        keep = [k and o is not None for k, o in zip(keep, other)]
    out = []
    for r, k in zip(rows, keep):
        if not k:
            continue
        c, t = r["codes"], r["text"]
        if c.shape[0] != cfg.n_streams:
            raise ValueError(f"{name}: {c.shape[0]} streams, config has {cfg.n_streams}")
        if c.shape[1] < 1 or c.shape[1] > cfg.max_frames or t.numel() < 1 or t.numel() > cfg.max_text:
            continue  # length limits (applied identically to every codec's copy)
        out.append({"codes": c.long(), "text": t.long()})
    return out


def collate(items: list[dict], cfg: TTSConfig) -> dict:
    """Left-padded text (+EOT), right-padded delayed audio."""
    texts = [torch.cat([it["text"], torch.tensor([EOT])]) for it in items]
    ys = [delayed_targets(it["codes"], cfg) for it in items]
    lt, la = max(len(t) for t in texts), max(y.shape[1] for y in ys)
    b, s_n = len(items), cfg.n_streams
    text = torch.full((b, lt), TEXT_PAD, dtype=torch.long)
    targets = torch.empty(b, s_n, la, dtype=torch.long)
    audio_in = torch.empty(b, s_n, la, dtype=torch.long)
    valid = torch.zeros(b, lt + la, dtype=torch.bool)
    for s, v in enumerate(cfg.vocab_sizes):
        targets[:, s].fill_(pad_id(v))
        audio_in[:, s].fill_(pad_id(v))
    for i, (t, y) in enumerate(zip(texts, ys)):
        text[i, lt - len(t):] = t
        n = y.shape[1]
        targets[i, :, :n] = y
        for s, v in enumerate(cfg.vocab_sizes):
            audio_in[i, s, 0] = bos(v)
            audio_in[i, s, 1:n] = y[s, : n - 1]
        valid[i, lt - len(t): lt] = True
        valid[i, lt: lt + n] = True
    return {"text": text, "audio_in": audio_in, "targets": targets, "valid": valid}


class TokenBatches(IterableDataset):
    """Infinite stream of collated batches (train) or one pass (val)."""

    def __init__(self, primary: Path, require: list[Path], names: list[str], cfg: TTSConfig,
                 max_tokens: int, rank: int = 0, world: int = 1, seed: int = 0,
                 infinite: bool = True, max_items: int = 256):
        self.primary, self.require, self.names = Path(primary), [Path(r) for r in require], names
        self.cfg, self.max_tokens, self.rank, self.world = cfg, max_tokens, rank, world
        self.seed, self.infinite, self.max_items = seed, infinite, max_items

    def _cost(self, it) -> int:
        return it["text"].numel() + 1 + it["codes"].shape[1] + self.cfg.max_delay + 1

    def __iter__(self):
        wi = get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        epoch = 0
        while True:
            rng = random.Random(self.seed + epoch)
            names = list(self.names)
            if self.infinite:
                rng.shuffle(names)
            mine = names[self.rank::self.world][wid::nw]
            for name in mine:
                items = load_items(name, self.primary, self.require, self.cfg)
                if self.infinite:
                    rng.shuffle(items)
                items.sort(key=self._cost)  # length-bucketed batches inside a shard
                batches, cur, cur_max = [], [], 0
                for it in items:
                    c = self._cost(it)
                    m = max(cur_max, c)
                    if cur and (m * (len(cur) + 1) > self.max_tokens or len(cur) >= self.max_items):
                        batches.append(cur)
                        cur, m = [], c
                    cur.append(it)
                    cur_max = m
                if cur:
                    batches.append(cur)
                if self.infinite:
                    rng.shuffle(batches)
                for b in batches:
                    yield collate(b, self.cfg)
            if not self.infinite:
                return
            epoch += 1
