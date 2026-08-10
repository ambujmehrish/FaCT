"""Download every model checkpoint into a cache OUTSIDE $HOME.

RUN ON A LOGIN NODE (compute nodes have no internet). CINECA $HOME is small
and not meant for model weights; this script refuses to write under $HOME
unless --allow-home is passed explicitly.

    export FACT_CACHE=$WORK/fact_cache        # add to your shell profile
    python scripts/download_checkpoints.py --baselines mimi,dac,xcodec2 \
        --whisper large-v3

Layout (everything self-contained under the cache dir):
    $FACT_CACHE/hf/        HF hub cache (mimi, xcodec2, ...) - jobs set
                           HF_HOME here and HF_HUB_OFFLINE=1
    $FACT_CACHE/dac/       DAC weights (the dac package would otherwise
                           cache under ~/.cache/descript - redirected here)
    $FACT_CACHE/whisper/   whisper checkpoints (via download_root)
    $FACT_CACHE/MANIFEST.json  what is present, with sizes

At job time the wrappers resolve the cache via the FACT_CACHE env var (see
slurm/*.sbatch); nothing touches $HOME or the network.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def resolve_cache_dir(arg: str | None, allow_home: bool) -> Path:
    cache = arg or os.environ.get("FACT_CACHE")
    if not cache:
        raise SystemExit(
            "No cache dir: pass --cache-dir or set FACT_CACHE "
            "(e.g. export FACT_CACHE=$WORK/fact_cache). Refusing to default "
            "to $HOME - compute nodes need a shared, quota-sized location."
        )
    path = Path(cache).expanduser().resolve()
    home = Path.home().resolve()
    if not allow_home and (path == home or home in path.parents):
        raise SystemExit(
            f"{path} is inside $HOME ({home}). CINECA $HOME is small and "
            f"slow for weights; use $WORK (or pass --allow-home to override)."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch_mimi(cache: Path) -> dict:
    from huggingface_hub import snapshot_download
    path = snapshot_download("kyutai/mimi")
    return {"path": str(path)}


def fetch_xcodec2(cache: Path) -> dict:
    from huggingface_hub import snapshot_download
    path = snapshot_download("HKUSTAudio/xcodec2")
    return {"path": str(path)}


def fetch_dac(cache: Path, model_type: str = "24khz") -> dict:
    import dac
    src = Path(dac.utils.download(model_type=model_type))  # ~/.cache/descript
    dst = cache / "dac" / f"weights_{model_type}.pth"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"path": str(dst)}


def fetch_whisper(cache: Path, size: str) -> dict:
    import whisper
    root = cache / "whisper"
    root.mkdir(parents=True, exist_ok=True)
    whisper.load_model(size, device="cpu", download_root=str(root))
    return {"path": str(root / f"{size}.pt")}


FETCHERS = {"mimi": fetch_mimi, "dac": fetch_dac, "xcodec2": fetch_xcodec2}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=None,
                    help="target cache (default: $FACT_CACHE; never $HOME)")
    ap.add_argument("--baselines", default="mimi,dac")
    ap.add_argument("--whisper", default=None,
                    help="also cache a whisper model (e.g. large-v3)")
    ap.add_argument("--allow-home", action="store_true")
    args = ap.parse_args()

    cache = resolve_cache_dir(args.cache_dir, args.allow_home)
    # HF must point at the cache BEFORE any hub import downloads anything.
    os.environ["HF_HOME"] = str(cache / "hf")
    os.environ.pop("HF_HUB_OFFLINE", None)  # this script is the online phase

    manifest_path = cache / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in [b.strip() for b in args.baselines.split(",") if b.strip()]:
        if name not in FETCHERS:
            raise SystemExit(f"Unknown baseline {name!r}; known: {sorted(FETCHERS)}")
        print(f"fetching {name} ...")
        manifest[name] = FETCHERS[name](cache)
        print(f"  {name} -> {manifest[name]['path']}")
    if args.whisper:
        print(f"fetching whisper {args.whisper} ...")
        manifest[f"whisper-{args.whisper}"] = fetch_whisper(cache, args.whisper)

    manifest_path.write_text(json.dumps(manifest, indent=2))
    total = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file())
    print(f"\nCache complete: {cache} ({total / 1e9:.2f} GB)")
    print("Job-side env (already in slurm/*.sbatch):")
    print(f"  export FACT_CACHE={cache}")
    print(f"  export HF_HOME={cache}/hf")
    print("  export HF_HUB_OFFLINE=1")


if __name__ == "__main__":
    main()
