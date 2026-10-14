"""Shard index caching and shard-grouped sampling."""

import json

import torch

from fact.config import tiny_config
from fact.data import dataset as dataset_mod
from fact.data.dataset import INDEX_NAME, ShardedSpeech, ShardGroupedSampler


def write_shards(root, sizes):
    root.mkdir(parents=True, exist_ok=True)
    cfg = tiny_config()
    for k, n in enumerate(sizes):
        items = [{"mel": torch.zeros(8, cfg.audio.n_mels, dtype=torch.float16),
                  "f0": torch.zeros(8), "text": torch.tensor([k + 1, i + 1])}
                 for i in range(n)]
        torch.save(items, root / f"shard_{k:06d}.pt")


def test_index_written_once_then_reused(tmp_path, monkeypatch):
    write_shards(tmp_path, [3, 5, 2])
    cfg = tiny_config()
    ds = ShardedSpeech(str(tmp_path), cfg)
    assert len(ds) == 10 and ds.counts == [3, 5, 2]
    assert json.loads((tmp_path / INDEX_NAME).read_text()) == {
        "shard_000000.pt": 3, "shard_000001.pt": 5, "shard_000002.pt": 2}

    def no_load(*a, **k):
        raise AssertionError("shards must not be loaded when index.json is present")
    monkeypatch.setattr(dataset_mod.torch, "load", no_load)
    assert ShardedSpeech(str(tmp_path), cfg).counts == [3, 5, 2]


def test_index_rebuilt_when_new_shard_appears(tmp_path):
    write_shards(tmp_path, [3, 5])
    cfg = tiny_config()
    ShardedSpeech(str(tmp_path), cfg)
    write_shards(tmp_path, [3, 5, 4])
    assert ShardedSpeech(str(tmp_path), cfg).counts == [3, 5, 4]


def test_getitem_maps_global_index_to_shard(tmp_path):
    write_shards(tmp_path, [3, 5, 2])
    ds = ShardedSpeech(str(tmp_path), tiny_config())
    # text = [shard + 1, item + 1]
    assert ds[0]["text"].tolist() == [1, 1]
    assert ds[3]["text"].tolist() == [2, 1]
    assert ds[9]["text"].tolist() == [3, 2]


def test_sampler_covers_epoch_and_groups_by_shard(tmp_path):
    write_shards(tmp_path, [4, 6, 3, 5])
    ds = ShardedSpeech(str(tmp_path), tiny_config())
    order = list(ShardGroupedSampler(ds, seed=7))
    assert sorted(order) == list(range(len(ds)))
    shard_of = [next(s for s in range(4) if i in ds.shard_range(s)) for i in order]
    switches = sum(a != b for a, b in zip(shard_of, shard_of[1:]))
    assert switches == 3  # each shard visited in one contiguous run


def test_sampler_ddp_split_is_disjoint_and_complete(tmp_path):
    write_shards(tmp_path, [4, 6, 3, 5, 2])
    ds = ShardedSpeech(str(tmp_path), tiny_config())
    parts = [list(ShardGroupedSampler(ds, seed=3, rank=r, world_size=2)) for r in range(2)]
    assert not set(parts[0]) & set(parts[1])
    assert sorted(parts[0] + parts[1]) == list(range(len(ds)))
    for r, part in enumerate(parts):
        assert len(ShardGroupedSampler(ds, seed=3, rank=r, world_size=2)) == len(part)


def test_sampler_epochs_differ_but_are_reproducible(tmp_path):
    write_shards(tmp_path, [4, 6, 3, 5])
    ds = ShardedSpeech(str(tmp_path), tiny_config())
    s = ShardGroupedSampler(ds, seed=1)
    e0 = list(s)
    s.set_epoch(1)
    e1 = list(s)
    assert e0 != e1
    s2 = ShardGroupedSampler(ds, seed=1)
    s2.set_epoch(1)
    assert list(s2) == e1


def test_compute_stats_writes_index(tmp_path):
    from fact.data.preprocess import compute_stats
    write_shards(tmp_path, [3, 2])
    cfg = tiny_config()
    compute_stats(tmp_path, cfg.audio.hop_length, cfg.audio.sample_rate)
    assert json.loads((tmp_path / INDEX_NAME).read_text()) == {
        "shard_000000.pt": 3, "shard_000001.pt": 2}
