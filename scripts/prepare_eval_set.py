"""Download a reconstruction eval set - RUN ON A LOGIN NODE (needs internet).

Streams LibriSpeech test-clean via HF datasets and writes 16 kHz wavs +
transcripts, deterministic order, so every baseline sees identical audio.

    pip install datasets soundfile
    python scripts/prepare_eval_set.py --out $WORK/data/eval/librispeech_test_clean -n 200

For the paper also prepare Emilia held-out / Expresso / ESD sets with
fact.data.manifest + your own copies (see docs/DATA.md); this script covers
the standard English set used to validate the environment against
published codec numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--dataset", default="openslr/librispeech_asr")
    ap.add_argument("--config", default="clean")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    import soundfile as sf
    from datasets import load_dataset

    args.out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    n = 0
    for ex in ds:
        audio = ex["audio"]
        stem = f"{n:04d}_{ex.get('id', 'utt')}"
        sf.write(args.out / f"{stem}.wav", audio["array"], audio["sampling_rate"])
        (args.out / f"{stem}.txt").write_text(ex.get("text", ""))
        n += 1
        if n >= args.n:
            break
    print(f"Wrote {n} utterances to {args.out}")


if __name__ == "__main__":
    main()
