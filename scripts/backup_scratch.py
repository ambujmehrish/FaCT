"""Copy everything a rebuttal may need that lives on SCRATCH (purged after
~40 days) to WORK. Anything already outside the scratch root is left in
place and listed as "already safe".

    python scripts/backup_scratch.py --out $FACT_BASE/rebuttal_backup/2026-11-10 \
        --scratch-root /leonardo_scratch --runs-dir $FACT_RUNS \
        --checkpoints ops/archive/rebuttal_checkpoints.txt \
        --manifest $FACT_DATA/manifests/dev.tsv --manifest $FACT_DATA/manifests/test.tsv \
        --dir $FACT_DATA/shards/dev --dir $FACT_CACHE ...

What is copied (full files, byte for byte; unlike scripts/archive_artifacts.py
nothing is stripped, so optimizer states allow resuming or fine-tuning):
- each checkpoint in --checkpoints (paths relative to --runs-dir) and the
  config files next to it (*.json, *.yaml);
- each --manifest (TSV, first column an audio path) and every audio file it
  references, plus a rewritten copy of the manifest pointing at the copies;
- each --dir, recursively (dev shards, model caches, ...).

Destination layout mirrors the source path below --scratch-root, so a path
maps back by replacing the prefix. Re-running skips files already copied
with the same size (resumable). A missing listed checkpoint is recorded under
"missing" in the manifest and makes the exit status non-zero; it is never
silently dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


class Backup:
    def __init__(self, out: Path, scratch_root: Path, hash_files: bool = True):
        self.out, self.root, self.hash = out, scratch_root.resolve(), hash_files
        self.copied, self.safe, self.missing, self.skipped = [], [], [], []
        self.bytes = 0

    def dest(self, src: Path) -> Path:
        return self.out / "scratch" / src.resolve().relative_to(self.root)

    def on_scratch(self, p: Path) -> bool:
        try:
            p.resolve().relative_to(self.root)
            return True
        except ValueError:
            return False

    def copy(self, src: Path, record_hash: bool = False) -> Path | None:
        if not src.exists():
            self.missing.append(str(src))
            return None
        if not self.on_scratch(src):
            self.safe.append(str(src))
            return src
        dst = self.dest(src)
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            self.skipped.append(str(src))
            return dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f"{dst.name}.{os.getpid()}.partial")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        entry = {"source": str(src), "copy": str(dst), "bytes": dst.stat().st_size}
        if record_hash and self.hash:
            entry["sha256"] = sha256(dst)
            if entry["sha256"] != sha256(src):
                raise SystemExit(f"checksum mismatch after copying {src}")
        self.copied.append(entry)
        self.bytes += entry["bytes"]
        return dst

    def copy_tree(self, d: Path) -> None:
        if not d.exists():
            self.missing.append(str(d))
            return
        if not self.on_scratch(d):
            self.safe.append(str(d))
            return
        for p in sorted(d.rglob("*")):
            if p.is_file():
                self.copy(p)


def backup_manifest(b: Backup, tsv: Path) -> None:
    if not tsv.exists():
        b.missing.append(str(tsv))
        return
    b.copy(tsv)
    lines, n_audio = [], 0
    for line in tsv.read_text(encoding="utf-8").splitlines():
        cols = line.split("\t")
        p = Path(cols[0])
        if cols[0] and p.is_absolute():
            new = b.copy(p)
            n_audio += 1
            if new is not None:
                cols[0] = str(new)
        lines.append("\t".join(cols))
    out = b.out / "manifests" / tsv.name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{tsv}: {n_audio} audio paths, rewritten copy -> {out}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scratch-root", type=Path, required=True)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--checkpoints", type=Path, required=True,
                    help="one checkpoint per line, relative to --runs-dir")
    ap.add_argument("--manifest", type=Path, action="append", default=[])
    ap.add_argument("--dir", type=Path, action="append", default=[])
    a = ap.parse_args(argv)
    if not a.scratch_root.is_dir():
        raise SystemExit(f"{a.scratch_root}: not a directory")
    if a.out.resolve().is_relative_to(a.scratch_root.resolve()):
        raise SystemExit(f"--out {a.out} is on scratch: it must be on WORK")
    rels = [ln.split("#", 1)[0].strip() for ln in a.checkpoints.read_text().splitlines()]
    rels = [r for r in rels if r]
    b = Backup(a.out, a.scratch_root)
    for i, rel in enumerate(rels):
        ck = a.runs_dir / rel
        b.copy(ck, record_hash=True)
        if ck.parent.is_dir():
            for side in sorted(ck.parent.glob("*.json")) + sorted(ck.parent.glob("*.yaml")):
                b.copy(side)
        print(f"[{i + 1}/{len(rels)}] {rel}", flush=True)
    for m in a.manifest:
        backup_manifest(b, m)
    for d in a.dir:
        b.copy_tree(d)
        print(f"dir {d} done", flush=True)
    from fact.provenance import provenance
    a.out.mkdir(parents=True, exist_ok=True)
    rep = {"scratch_root": str(a.scratch_root), "runs_dir": str(a.runs_dir),
           "copied_bytes_this_run": b.bytes, "copied": b.copied,
           "already_safe": sorted(set(b.safe)), "skipped_existing": len(b.skipped),
           "missing": b.missing, "provenance": provenance()}
    (a.out / "backup_manifest.json").write_text(json.dumps(rep, indent=1))
    print(f"copied {len(b.copied)} files ({b.bytes / 1e9:.1f} GB), already safe {len(set(b.safe))}, "
          f"skipped existing {len(b.skipped)}, missing {len(b.missing)}", flush=True)
    for m in b.missing:
        print(f"MISSING {m}", flush=True)
    return 1 if b.missing else 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.exit(main())
