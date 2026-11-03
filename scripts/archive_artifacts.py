"""Archive what the reported results need, off SCRATCH (purged after ~40 days).

    python scripts/archive_artifacts.py --runs-dir <runs> --out <WORK>/fact/archive/<date> \
        --list ops/archive/checkpoints.txt

For each checkpoint in --list (paths relative to --runs-dir), writes
<out>/checkpoints/<same relative path> with the inference weights and the
settings that produced them (model/generator weights, config, run, step and
the stage-D fields). Optimizer and discriminator states are dropped, which
makes the files about 3x smaller.

Also writes:
- manifest.json: per file, the source path, size and sha256, the archived path
  and sha256, and the kept/dropped keys;
- env/pip_freeze.txt, env/python.txt;
- env/hf_revisions.json: the snapshot commit of every Hugging Face model in
  the cache, pinning the exact revisions the evaluations loaded.

A missing checkpoint is an error, never skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

KEEP = ("model", "generator", "config", "run", "step", "fact_ckpt", "in_dim", "residual_drop")


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def strip(src: Path, dst: Path) -> dict:
    ck = torch.load(src, map_location="cpu", weights_only=True)
    if not isinstance(ck, dict) or not ({"model", "generator"} & set(ck)):
        raise SystemExit(f"{src}: not a FaCT/stage-D/TTS checkpoint (keys {sorted(ck)[:8]})")
    kept = {k: ck[k] for k in KEEP if k in ck}
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.{os.getpid()}.partial")
    torch.save(kept, tmp)
    os.replace(tmp, dst)
    return {"kept": sorted(kept), "dropped": sorted(set(ck) - set(kept))}


def hf_revisions(roots: list[Path]) -> dict:
    out = {}
    for root in roots:
        hub = root / "hub" if (root / "hub").is_dir() else root
        for repo in sorted(hub.glob("models--*")):
            name = repo.name[len("models--"):].replace("--", "/")
            refs = {r.name: r.read_text().strip() for r in (repo / "refs").glob("*")} if (repo / "refs").is_dir() else {}
            snaps = sorted(p.name for p in (repo / "snapshots").glob("*")) if (repo / "snapshots").is_dir() else []
            out[name] = {"cache": str(repo), "refs": refs, "snapshots": snaps}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--list", type=Path, required=True, help="one checkpoint path per line, relative to --runs-dir")
    a = ap.parse_args(argv)
    rels = [ln.split("#", 1)[0].strip() for ln in a.list.read_text(encoding="utf-8").splitlines()]
    rels = [r for r in rels if r]
    missing = [r for r in rels if not (a.runs_dir / r).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} checkpoints missing under {a.runs_dir}: {missing[:5]}")
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = {"runs_dir": str(a.runs_dir), "files": []}
    for i, rel in enumerate(rels):
        src, dst = a.runs_dir / rel, a.out / "checkpoints" / rel
        info = strip(src, dst)
        manifest["files"].append({"source": str(src), "source_bytes": src.stat().st_size,
                                  "source_sha256": sha256(src), "archived": str(dst),
                                  "archived_bytes": dst.stat().st_size, "archived_sha256": sha256(dst), **info})
        print(f"[{i + 1}/{len(rels)}] {rel}: {src.stat().st_size / 1e9:.2f} GB -> {dst.stat().st_size / 1e9:.2f} GB",
              flush=True)
    env = a.out / "env"
    env.mkdir(exist_ok=True)
    (env / "pip_freeze.txt").write_text(
        subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout)
    (env / "python.txt").write_text(f"{sys.executable}\n{sys.version}\ntorch {torch.__version__}\n"
                                    f"cuda {torch.version.cuda}\n")
    roots = [Path(p) for p in {os.environ.get("HF_HOME"), os.environ.get("HF_HUB_CACHE"),
                               os.environ.get("FACT_CACHE")} if p]
    (env / "hf_revisions.json").write_text(json.dumps(hf_revisions(roots), indent=1))
    from fact.provenance import provenance
    manifest["provenance"] = provenance()
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"archived {len(rels)} checkpoints to {a.out}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
