"""Pre-download baseline checkpoints - RUN ON A LOGIN NODE (needs internet).

CINECA compute nodes are offline: set HF_HOME to a shared path (e.g.
$WORK/hf_cache), run this once, then submit jobs with HF_HUB_OFFLINE=1.

    export HF_HOME=$WORK/hf_cache
    python scripts/download_checkpoints.py --baselines mimi,dac
"""

from __future__ import annotations

import argparse


def fetch_mimi() -> None:
    from huggingface_hub import snapshot_download
    path = snapshot_download("kyutai/mimi")
    print(f"mimi -> {path}")


def fetch_dac() -> None:
    import dac
    path = dac.utils.download(model_type="24khz")
    print(f"dac 24khz -> {path}")


def fetch_xcodec2() -> None:
    from huggingface_hub import snapshot_download
    path = snapshot_download("HKUSTAudio/xcodec2")
    print(f"xcodec2 -> {path}")


def fetch_whisper(size: str = "large-v3") -> None:
    import whisper
    whisper.load_model(size, device="cpu")
    print(f"whisper {size} cached")


FETCHERS = {"mimi": fetch_mimi, "dac": fetch_dac, "xcodec2": fetch_xcodec2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default="mimi,dac")
    ap.add_argument("--whisper", default=None, help="also cache whisper (e.g. large-v3)")
    args = ap.parse_args()
    for name in args.baselines.split(","):
        FETCHERS[name.strip()]()
    if args.whisper:
        fetch_whisper(args.whisper)


if __name__ == "__main__":
    main()
