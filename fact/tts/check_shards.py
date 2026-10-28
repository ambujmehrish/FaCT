"""Check that every token shard of the compared codecs loads, and that the
codecs agree on each shard's item count.

    python -m fact.tts.check_shards --dirs <tok>/fact_b1 <tok>/mimi2_b1 \
        --out results/tts/shard_check.json --quarantine <tok>/quarantine

An unreadable shard (e.g. truncated by a node failure after its write) is
moved to --quarantine/<dir name>/ (never deleted) so that a tokenize rerun,
which skips existing shards, writes it again. Count mismatches are only
reported: they mean different source shards, which a rerun cannot fix.
Exit status 1 if any count mismatch is found.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


def try_load(path: Path) -> tuple[int | None, str | None]:
    try:
        rows = torch.load(path, map_location="cpu", weights_only=True)
        return len(rows), None
    except Exception as e:  # recorded per file, nothing is skipped silently
        return None, f"{type(e).__name__}: {str(e)[:200]}"


def check(dirs: list[Path], threads: int = 32) -> dict:
    names = sorted({p.name for d in dirs for p in d.glob("shard_*.pt")})
    jobs = [(d, n) for d in dirs for n in names if (d / n).exists()]
    with ThreadPoolExecutor(threads) as ex:
        res = dict(zip(jobs, ex.map(lambda j: try_load(j[0] / j[1]), jobs)))
    bad = {str(d / n): err for (d, n), (_, err) in res.items() if err}
    missing = {str(d): [n for n in names if not (d / n).exists()] for d in dirs}
    mismatch = {}
    for n in names:
        counts = {str(d): res[(d, n)][0] for d in dirs if (d, n) in res and res[(d, n)][0] is not None}
        if len(set(counts.values())) > 1:
            mismatch[n] = counts
    return {"dirs": [str(d) for d in dirs], "shards": len(names), "files_checked": len(jobs),
            "unreadable": bad, "missing": {k: v for k, v in missing.items() if v},
            "count_mismatch": mismatch}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m fact.tts.check_shards")
    ap.add_argument("--dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quarantine", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args(argv)
    for d in a.dirs:
        if not (d / "meta.json").exists():
            ap.error(f"{d}/meta.json missing")
    rep = check(a.dirs, a.threads)
    moved = []
    if a.quarantine is not None:
        for p in rep["unreadable"]:
            src = Path(p)
            dst = a.quarantine / src.parent.name / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            moved.append(str(dst))
    rep["quarantined"] = moved
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(f"{a.out.name}.{os.getpid()}.partial")
    tmp.write_text(json.dumps(rep, indent=1))
    os.replace(tmp, a.out)
    print(json.dumps({k: (len(v) if isinstance(v, (dict, list)) else v) for k, v in rep.items()}, indent=1))
    if rep["count_mismatch"]:
        raise SystemExit("count mismatch between codecs (see report)")


if __name__ == "__main__":
    main()
