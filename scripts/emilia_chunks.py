"""Chunked Emilia staging: fit a 2.5 TB corpus through a ~1 TB $WORK quota.

Raw Emilia is ~25 MB per audio-hour; shards are ~37 MB/h (fp16) or ~19 MB/h
(int8). Nothing requires the full raw corpus on disk: this driver cycles
    stage (login node, internet)  ->  preprocess job  ->  verify-and-clean
so the peak footprint is  shards-so-far + one chunk of raw audio.

    # LOGIN NODE: download + extract the next ~100 GB of tars, write the
    # chunk manifest (state in <dest>/state.json; rerun to stage the next chunk)
    python scripts/emilia_chunks.py stage --lang EN \
        --dest $FACT_DATA/data/raw/emilia --chunk-gb 100

    # JOB: preprocess that chunk, appending into ONE shard dir
    MANIFEST=$FACT_DATA/data/raw/emilia/manifests/chunk_000.tsv \
        OUT=$FACT_DATA/shards/train CONFIG=configs/fact_base.yaml \
        EXTRA="--shard-prefix c000_ --skip-stats --mel-dtype int8" \
        sbatch slurm/preprocess.sbatch

    # LOGIN NODE: verify the chunk's shards exist, then delete its raw audio
    python scripts/emilia_chunks.py verify-and-clean --chunk 0 \
        --dest $FACT_DATA/data/raw/emilia --shards $FACT_DATA/shards/train

    # ... repeat until the hour budget is met (status shows progress), then:
    python -m fact.data.preprocess --out $FACT_DATA/shards/train --stats-only

Requires `huggingface_hub` and an accepted Emilia gate (`huggingface-cli
login`) for the stage command; verify-and-clean and status are offline.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ID = "amphion/Emilia-Dataset"
SHARD_SIZE_DEFAULT = 512  # must match the preprocess job's --shard-size


def load_state(dest: Path) -> dict:
    p = dest / "state.json"
    return json.loads(p.read_text()) if p.exists() else {"tars": {}, "chunks": {}}


def save_state(dest: Path, state: dict) -> None:
    (dest / "state.json").write_text(json.dumps(state, indent=2))


def list_remote_tars(lang: str) -> list[dict]:
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    # Emilia layout: Emilia/<LANG>/*.tar (older mirrors: <LANG>/*.tar)
    tars = sorted(f for f in files if f.endswith(".tar")
                  and (f.startswith(f"Emilia/{lang}/") or f.startswith(f"{lang}/")))
    if not tars:
        raise SystemExit(f"No tars found for language {lang!r} in {REPO_ID}")
    return tars


def cmd_stage(args) -> None:
    from huggingface_hub import hf_hub_download
    dest = Path(args.dest)
    (dest / "manifests").mkdir(parents=True, exist_ok=True)
    state = load_state(dest)

    done = set(state["tars"])
    todo = [t for t in list_remote_tars(args.lang) if t not in done]
    if not todo:
        print("All tars already staged/processed for this language.")
        return
    chunk_id = len(state["chunks"])
    chunk_key = f"{chunk_id:03d}"
    chunk_dir = dest / "audio" / f"chunk_{chunk_key}"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    budget = args.chunk_gb * 1e9
    used = 0.0
    staged: list[str] = []
    for tar_name in todo:
        if used >= budget:
            break
        print(f"downloading {tar_name} ...")
        local = Path(hf_hub_download(REPO_ID, tar_name, repo_type="dataset"))
        size = local.stat().st_size
        with tarfile.open(local) as tf:
            tf.extractall(chunk_dir, filter="data")  # refuses path traversal
        local.unlink()  # tar itself is not needed once extracted
        used += size
        staged.append(tar_name)
        state["tars"][tar_name] = {"chunk": chunk_key, "gb": round(size / 1e9, 2)}
        save_state(dest, state)
    if not staged:
        print("Nothing staged (budget too small for one tar?)")
        return

    from fact.data.manifest import scan_emilia, write_manifest
    manifest = dest / "manifests" / f"chunk_{chunk_key}.tsv"
    kept, skipped = write_manifest(scan_emilia(chunk_dir), manifest,
                                   require_text=True)
    state["chunks"][chunk_key] = {
        "tars": staged, "manifest": str(manifest), "utterances": kept,
        "status": "staged",
    }
    save_state(dest, state)
    print(f"\nchunk {chunk_key}: {len(staged)} tars, {used / 1e9:.1f} GB raw, "
          f"{kept} utterances ({skipped} without text skipped)")
    print(f"next: MANIFEST={manifest} OUT=<shard_dir> "
          f"EXTRA=\"--shard-prefix c{chunk_key}_ --skip-stats\" "
          f"sbatch slurm/preprocess.sbatch")


def cmd_verify_and_clean(args) -> None:
    dest = Path(args.dest)
    state = load_state(dest)
    chunk_key = f"{int(args.chunk):03d}"
    chunk = state["chunks"].get(chunk_key)
    if chunk is None:
        raise SystemExit(f"Unknown chunk {chunk_key}; see status")
    expected = math.ceil(chunk["utterances"] / args.shard_size)
    shards = list(Path(args.shards).glob(f"shard_c{chunk_key}_*.pt"))
    if len(shards) < expected and not args.force:
        raise SystemExit(
            f"chunk {chunk_key}: found {len(shards)} shards, expected "
            f"~{expected} ({chunk['utterances']} utts / {args.shard_size}). "
            f"Preprocess incomplete - NOT deleting raw audio (--force to "
            f"override if heavy filtering explains the gap; check drops.log)."
        )
    raw_dir = dest / "audio" / f"chunk_{chunk_key}"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    chunk["status"] = "cleaned"
    save_state(dest, state)
    print(f"chunk {chunk_key}: {len(shards)} shards verified, raw audio deleted.")


def cmd_status(args) -> None:
    dest = Path(args.dest)
    state = load_state(dest)
    total_gb = sum(t["gb"] for t in state["tars"].values())
    total_utts = sum(c["utterances"] for c in state["chunks"].values())
    print(f"{len(state['chunks'])} chunks, {len(state['tars'])} tars, "
          f"{total_gb:.0f} GB raw downloaded, ~{total_utts} utterances "
          f"(~{total_utts * 9 / 3600:.0f} h at ~9 s/utt)")
    for k, c in sorted(state["chunks"].items()):
        print(f"  chunk {k}: {c['status']:8s} {len(c['tars'])} tars, "
              f"{c['utterances']} utts")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("stage", help="download+extract next chunk (login node)")
    st.add_argument("--lang", required=True)
    st.add_argument("--dest", required=True)
    st.add_argument("--chunk-gb", type=float, default=100.0)
    vc = sub.add_parser("verify-and-clean", help="delete a chunk's raw audio once sharded")
    vc.add_argument("--chunk", required=True)
    vc.add_argument("--dest", required=True)
    vc.add_argument("--shards", required=True)
    vc.add_argument("--shard-size", type=int, default=SHARD_SIZE_DEFAULT)
    vc.add_argument("--force", action="store_true")
    ss = sub.add_parser("status")
    ss.add_argument("--dest", required=True)
    args = ap.parse_args()
    {"stage": cmd_stage, "verify-and-clean": cmd_verify_and_clean,
     "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()
