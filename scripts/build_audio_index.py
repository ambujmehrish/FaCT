"""Map every preprocessed shard item back to its audio file (for #4, the
SSL-input run, which needs the waveform the shards do not store).

    python scripts/build_audio_index.py --shards <shards>/train_v2 \
        --manifests '<manifests>/train_c[0-9][0-9][0-9].tsv'
    python scripts/build_audio_index.py --shards <shards>/dev \
        --manifests <manifests>/dev.tsv --no-prefix

Preprocessing (fact/data/preprocess.py) turns manifest lines
[gi*512, (gi+1)*512) of a chunk into shard_<prefix><gi:06d>.pt, keeping
input order and dropping files that were filtered or failed - each drop is
written to drops.log with its path. So a shard's items are exactly its
group's lines minus the dropped paths. That is CHECKED against the shard's
item count in index.json for every shard (any mismatch aborts), and again
per item at training time (decoded audio length vs stored mel length).

Writes <shards>/audio_index/<shard stem>.txt: one audio path per item.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

FILTERED = re.compile(r"^shard (\d+): filtered\([^)]*\): (.*)$")
ERROR = re.compile(r"^shard (\d+): error\(")


def parse_drops(drops_log: Path) -> tuple[set[str], dict[int, list[str]]]:
    """(paths of duration-filtered files, error lines by shard index).

    Error lines end with the manifest line cut at 120 characters, and the
    exception text before it may contain anything, so a failed file is
    identified by searching its own path in the error lines logged for its
    shard index (shard indices repeat across chunks: a few lines each)."""
    filtered: set[str] = set()
    errors: dict[int, list[str]] = {}
    if not drops_log.exists():
        return filtered, errors
    for ln in drops_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if not ln.strip():
            continue
        m = FILTERED.match(ln)
        if m:
            filtered.add(m.group(2).strip())
            continue
        m = ERROR.match(ln)
        if not m:
            raise ValueError(f"unrecognised drops.log line: {ln[:160]}")
        errors.setdefault(int(m.group(1)), []).append(ln)
    return filtered, errors


def build(shards: Path, manifests: list[Path], prefixed: bool, shard_size: int,
          out: Path) -> dict:
    counts = json.loads((shards / "index.json").read_text())
    filtered, errors = parse_drops(shards / "drops.log")
    out.mkdir(parents=True, exist_ok=True)
    n_items = n_shards = 0
    seen = set()
    for man in manifests:
        tag = re.sub(r"^train_", "", man.stem) + "_" if prefixed else ""
        with open(man, encoding="utf-8") as fh:
            lines = [ln.rstrip("\n") for ln in fh if ln.strip()]  # as preprocess.py
        for gi in range(0, -(-len(lines) // shard_size)):
            name = f"shard_{tag}{gi:06d}.pt"
            paths = [ln.split("\t", 1)[0] for ln in lines[gi * shard_size:(gi + 1) * shard_size]]
            errs = errors.get(gi, [])
            keep = [p for p in paths
                    if p not in filtered and not any(p[:120] in e for e in errs)]
            if name not in counts:
                if keep:
                    raise SystemExit(f"{name}: missing from index.json but {len(keep)} items expected")
                continue  # every file of the group dropped -> no shard written
            if len(keep) != counts[name]:
                raise SystemExit(f"{name}: {counts[name]} items in index.json, {len(keep)} "
                                 f"after removing drops from {man.name} lines "
                                 f"{gi * shard_size}-{(gi + 1) * shard_size - 1}")
            (out / f"{Path(name).stem}.txt").write_text("\n".join(keep) + "\n", encoding="utf-8")
            seen.add(name)
            n_items += len(keep)
            n_shards += 1
    missing = sorted(set(counts) - seen)
    if missing:
        raise SystemExit(f"{len(missing)} shards in index.json not produced by these manifests "
                         f"(e.g. {missing[:3]})")
    report = {"shards": n_shards, "items": n_items, "dropped_filtered": len(filtered),
              "dropped_errors": sum(len(v) for v in errors.values())}
    (out / "_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--manifests", required=True, help="file or glob, in chunk order")
    ap.add_argument("--no-prefix", action="store_true",
                    help="unchunked preprocessing (shard_<gi>.pt, e.g. dev)")
    ap.add_argument("--shard-size", type=int, default=512)
    args = ap.parse_args()
    mans = [Path(p) for p in sorted(glob.glob(args.manifests))]
    if not mans:
        raise SystemExit(f"no manifests match {args.manifests}")
    print(json.dumps(build(args.shards, mans, not args.no_prefix, args.shard_size,
                           args.shards / "audio_index"), indent=2))


if __name__ == "__main__":
    main()
