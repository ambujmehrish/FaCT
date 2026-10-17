"""Group-disjoint train/dev/test split of the Emilia manifest + preprocessing chunks.

    python scripts/split_emilia.py --manifest <manifests>/emilia_en.tsv \
        --out-dir <manifests> --chunks 36

Emilia segment ids are LANG_Bxxxxx_Sxxxxx_Wxxxxxx (batch, group, segment).
The split unit is the (B, S) group: every segment of a held-out group goes
to held-out, none to train, so no dev/test group is ever seen in training.
That is the finest identity Emilia exposes above the segment; a person who
appears under several groups (e.g. a recurring podcast host) cannot be
detected from ids alone - state that in the paper.

Held-out groups are chosen by a stable hash of the group id (no RNG state,
identical on every machine). dev/test are sampled from them with at most
--per-group utterances per group, so they cover many groups rather than a
few long ones.

Writes, in --out-dir:
  train.tsv            all utterances of non-held-out groups
  heldout.tsv          all utterances of held-out groups
  dev.tsv, test.tsv    sampled from held-out; each held-out group feeds only
                       one of them, so they are group-disjoint too
  train_cNNN.tsv       --chunks pieces of the SHUFFLED train.tsv (preprocessing;
                       shards must be random samples, see shuffle_and_chunk)
  split_report.json    counts + group-size stats + overlap self-check

Reads the 4.6 GB manifest line by line; run it as a batch job
(slurm/split_emilia.sbatch), not on a login node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def group_id(path: str) -> str:
    """'.../EN_B00054_S01894_W000010.mp3' -> 'EN_B00054_S01894'."""
    stem = Path(path).stem
    parts = stem.split("_")
    if len(parts) != 4 or not parts[1].startswith("B") or not parts[2].startswith("S") \
            or not parts[3].startswith("W"):
        raise ValueError(f"not an Emilia segment id: {stem!r}")
    return "_".join(parts[:3])


def stable_bucket(key: str, n: int = 10_000) -> int:
    return int.from_bytes(hashlib.md5(key.encode()).digest()[:8], "big") % n


def shuffle_and_chunk(out: Path, n_chunks: int, seed: int) -> list[str]:
    """Shuffle train.tsv in place (fixed seed) and cut it into n_chunks pieces.

    The sampler batches within a shard and visits one shard at a time, which
    is only random if every shard is a random sample of the corpus. The
    manifest is in directory order (consecutive lines share Emilia groups and
    recordings), so train.tsv must be shuffled before chunking/preprocessing.
    """
    with open(out / "train.tsv", encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()]
    random.Random(seed).shuffle(lines)
    tmp = out / "train.tsv.tmp"
    tmp.write_text("".join(lines), encoding="utf-8")
    tmp.rename(out / "train.tsv")
    for old in out.glob("train_c[0-9][0-9][0-9].tsv"):
        old.unlink()
    per_chunk = -(-len(lines) // n_chunks)
    names = []
    for ci in range(0, len(lines), per_chunk):
        name = f"train_c{ci // per_chunk:03d}.tsv"
        (out / name).write_text("".join(lines[ci : ci + per_chunk]), encoding="utf-8")
        names.append(name)
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--heldout-per-10k", type=int, default=100,
                    help="held-out groups per 10,000 (default 1%%; Emilia groups are "
                         "large, so 0.3%% leaves too few groups for dev+test)")
    ap.add_argument("--n-dev", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=5000)
    ap.add_argument("--per-group", type=int, default=3)
    ap.add_argument("--chunks", type=int, default=36)
    ap.add_argument("--shuffle-seed", type=int, default=20261017)
    ap.add_argument("--reshuffle-train", action="store_true",
                    help="existing split: shuffle train.tsv and rewrite the chunks only "
                         "(dev/test/heldout untouched)")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.reshuffle_train:
        rep_path = out / "split_report.json"
        if not (out / "train.tsv").exists() or not rep_path.exists():
            raise SystemExit(f"--reshuffle-train needs an existing split in {out}")
        names = shuffle_and_chunk(out, args.chunks, args.shuffle_seed)
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        rep["chunks"] = names
        rep["train_shuffle_seed"] = args.shuffle_seed
        rep_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"shuffled train.tsv (seed {args.shuffle_seed}) into {len(names)} chunks")
        return
    for name in ("train.tsv", "heldout.tsv", "dev.tsv", "test.tsv"):
        if (out / name).exists():
            raise SystemExit(f"{out / name} exists; refusing to overwrite a split "
                             "that shards may already be built from")

    group_sizes: Counter[str] = Counter()
    held_groups: set[str] = set()
    n_train = n_held = 0
    with open(args.manifest, encoding="utf-8") as fin, \
            open(out / "train.tsv.tmp", "w", encoding="utf-8") as ftr, \
            open(out / "heldout.tsv.tmp", "w", encoding="utf-8") as fho:
        for line in fin:
            if not line.strip():
                continue
            g = group_id(line.split("\t", 1)[0])
            group_sizes[g] += 1
            if stable_bucket(g) < args.heldout_per_10k:
                held_groups.add(g)
                fho.write(line)
                n_held += 1
            else:
                ftr.write(line)
                n_train += 1

    # dev/test: walk held-out in a stable hashed order, <= per_group each.
    with open(out / "heldout.tsv.tmp", encoding="utf-8") as fh:
        held = [ln for ln in fh if ln.strip()]
    held.sort(key=lambda ln: stable_bucket(ln.split("\t", 1)[0], 1 << 62))
    # Each held-out group belongs to exactly one of dev/test (by hash), so
    # dev and test are group-disjoint from each other too.
    taken: Counter[str] = Counter()
    dev, test = [], []
    for ln in held:
        g = group_id(ln.split("\t", 1)[0])
        if taken[g] >= args.per_group:
            continue
        target, cap = (test, args.n_test) if stable_bucket(g + "#split", 2) == 0 \
            else (dev, args.n_dev)
        if len(target) < cap:
            target.append(ln)
            taken[g] += 1
        if len(test) >= args.n_test and len(dev) >= args.n_dev:
            break
    if len(test) < args.n_test or len(dev) < args.n_dev:
        need = -(-args.n_test // args.per_group) * 2  # test takes ~half the groups
        raise SystemExit(
            f"held-out too small: {len(held_groups)} of {len(group_sizes)} groups "
            f"({n_held} utterances) gave test={len(test)}/{args.n_test}, "
            f"dev={len(dev)}/{args.n_dev} at per_group={args.per_group}. Needs ~{need} "
            f"held-out groups: --heldout-per-10k >= "
            f"{-(-need * 10_000 // max(1, len(group_sizes)))}")

    # Self-check before anything is published: no held-out group in train.
    test_groups = {group_id(ln.split("\t", 1)[0]) for ln in test}
    dev_groups = {group_id(ln.split("\t", 1)[0]) for ln in dev}
    leaks = 0
    with open(out / "train.tsv.tmp", encoding="utf-8") as fh:
        for ln in fh:
            if group_id(ln.split("\t", 1)[0]) in held_groups:
                leaks += 1
    if leaks:
        raise SystemExit(f"BUG: {leaks} train utterances belong to held-out groups")

    (out / "dev.tsv").write_text("".join(dev), encoding="utf-8")
    (out / "test.tsv").write_text("".join(test), encoding="utf-8")
    (out / "heldout.tsv.tmp").rename(out / "heldout.tsv")
    (out / "train.tsv.tmp").rename(out / "train.tsv")

    chunk_files = shuffle_and_chunk(out, args.chunks, args.shuffle_seed)

    sizes = sorted(group_sizes.values())
    report = {
        "manifest": str(args.manifest),
        "unit": "Emilia (B, S) group",
        "n_utterances": n_train + n_held,
        "n_groups": len(group_sizes),
        "group_size": {"median": sizes[len(sizes) // 2], "p99": sizes[int(0.99 * (len(sizes) - 1))],
                       "max": sizes[-1]},
        "train": {"utterances": n_train, "groups": len(group_sizes) - len(held_groups)},
        "heldout": {"utterances": n_held, "groups": len(held_groups)},
        "dev": {"utterances": len(dev), "groups": len(dev_groups)},
        "test": {"utterances": len(test), "groups": len(test_groups)},
        "dev_test_group_overlap": len(dev_groups & test_groups),
        "train_heldout_group_overlap": leaks,
        "chunks": chunk_files,
        "train_shuffle_seed": args.shuffle_seed,
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (out / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "chunks"}, indent=2))


if __name__ == "__main__":
    main()
