"""Build preprocessing manifests (TSV: wav_path<TAB>transcript) from common
dataset layouts.

Supported layouts:
  libritts   LibriTTS / LibriTTS-R trees: <root>/**/<utt>.wav with a sibling
             <utt>.normalized.txt (preferred) or <utt>.original.txt.
  emilia     Extracted Emilia webdataset shards: <root>/**/<utt>.mp3 with a
             sibling <utt>.json containing a "text" field.
  jsonl      A .jsonl file; keys configurable (--audio-key/--text-key), with
             relative audio paths resolved against --root.
  generic    Any audio tree (<root>/**/*.{wav,flac,mp3}); transcript from a
             sibling .txt/.lab file when present, else empty (audio-only
             utterances still train everything except CTC).

Usage:
  python -m fact.data.manifest libritts --root data/LibriTTS_R --out train.tsv
  python -m fact.data.manifest emilia --root data/emilia/EN --out emilia_en.tsv
  python -m fact.data.manifest jsonl --root data/ --jsonl meta.jsonl --out m.tsv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".opus")


def _clean(text: str) -> str:
    return " ".join(text.split())


def _sibling_text(audio: Path, suffixes: list[str]) -> str:
    for suf in suffixes:
        p = audio.with_suffix(suf)
        if p.exists():
            return _clean(p.read_text(encoding="utf-8", errors="replace"))
    return ""


def scan_libritts(root: Path) -> list[tuple[str, str]]:
    items = []
    for wav in sorted(root.rglob("*.wav")):
        text = _sibling_text(wav, [".normalized.txt", ".original.txt", ".txt"])
        items.append((str(wav), text))
    return items


def scan_emilia(root: Path) -> list[tuple[str, str]]:
    items = []
    for audio in sorted(root.rglob("*")):
        if audio.suffix.lower() not in AUDIO_EXTS:
            continue
        meta = audio.with_suffix(".json")
        text = ""
        if meta.exists():
            try:
                text = _clean(json.loads(meta.read_text(encoding="utf-8")).get("text", ""))
            except json.JSONDecodeError:
                pass
        items.append((str(audio), text))
    return items


def scan_generic(root: Path) -> list[tuple[str, str]]:
    items = []
    for audio in sorted(root.rglob("*")):
        if audio.suffix.lower() not in AUDIO_EXTS:
            continue
        items.append((str(audio), _sibling_text(audio, [".txt", ".lab"])))
    return items


def scan_jsonl(jsonl: Path, root: Path | None,
               audio_key: str = "wav", text_key: str = "text") -> list[tuple[str, str]]:
    items = []
    with open(jsonl, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            audio = Path(entry[audio_key])
            if root is not None and not audio.is_absolute():
                audio = root / audio
            items.append((str(audio), _clean(str(entry.get(text_key, "")))))
    return items


def write_manifest(items: list[tuple[str, str]], out: Path,
                   require_text: bool = False) -> tuple[int, int]:
    kept = skipped = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for path, text in items:
            if require_text and not text:
                skipped += 1
                continue
            fh.write(f"{path}\t{text}\n")
            kept += 1
    return kept, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("layout", choices=["libritts", "emilia", "jsonl", "generic"])
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--jsonl", type=Path, default=None)
    ap.add_argument("--audio-key", default="wav")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--require-text", action="store_true",
                    help="drop utterances without a transcript")
    args = ap.parse_args()

    if args.layout == "jsonl":
        if args.jsonl is None:
            ap.error("jsonl layout requires --jsonl")
        items = scan_jsonl(args.jsonl, args.root, args.audio_key, args.text_key)
    else:
        if args.root is None:
            ap.error(f"{args.layout} layout requires --root")
        items = {"libritts": scan_libritts, "emilia": scan_emilia,
                 "generic": scan_generic}[args.layout](args.root)

    kept, skipped = write_manifest(items, args.out, args.require_text)
    print(f"Wrote {kept} utterances to {args.out}"
          + (f" ({skipped} skipped: no transcript)" if skipped else ""))


if __name__ == "__main__":
    main()
