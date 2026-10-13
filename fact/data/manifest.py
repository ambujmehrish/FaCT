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
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _scan_emilia_dir(subdir: str) -> list[tuple[str, str]]:
    """One extracted Emilia tar directory. Pairs audio with its .json sidecar
    from the directory listing instead of a stat per file (Lustre metadata
    calls are the bottleneck)."""
    items = []
    for dirpath, _, files in os.walk(subdir):
        names = set(files)
        for f in sorted(files):
            stem, ext = os.path.splitext(f)
            if ext.lower() not in AUDIO_EXTS:
                continue
            text = ""
            if stem + ".json" in names:
                try:
                    with open(os.path.join(dirpath, stem + ".json"), encoding="utf-8") as fh:
                        text = _clean(str(json.load(fh).get("text", "")))
                except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                    pass
            items.append((os.path.join(dirpath, f), text))
    return items


def scan_emilia_parallel(root: Path, out: Path, workers: int,
                         require_text: bool = False) -> tuple[int, int]:
    """Scan each top-level subdirectory of `root` in its own process.

    Each subdirectory's result is written to `<out>.parts/<name>.tsv` as soon
    as it finishes, so a killed or timed-out run resumes where it stopped.
    The final manifest is assembled only after every part exists.
    """
    subdirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.endswith(".extracting"))
    if not subdirs:
        return write_manifest(scan_emilia(root), out, require_text)
    parts = out.parent / (out.name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    todo = [d for d in subdirs if not (parts / f"{d.name}.tsv").exists()]
    print(f"{len(subdirs)} directories, {len(subdirs) - len(todo)} already scanned, "
          f"{len(todo)} to go ({workers} workers)", flush=True)

    done = len(subdirs) - len(todo)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_scan_emilia_dir, str(d)): d for d in todo}
        for fut in as_completed(futures):
            d = futures[fut]
            items = fut.result()
            tmp = parts / f"{d.name}.tsv.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.writelines(f"{p}\t{t}\n" for p, t in items)
            tmp.rename(parts / f"{d.name}.tsv")
            done += 1
            print(f"[{done}/{len(subdirs)}] {d.name}: {len(items)} utterances", flush=True)

    kept = skipped = 0
    tmp_out = out.with_name(out.name + ".tmp")
    with open(tmp_out, "w", encoding="utf-8") as fh:
        for d in subdirs:
            with open(parts / f"{d.name}.tsv", encoding="utf-8") as part:
                for line in part:
                    if require_text and not line.rstrip("\n").split("\t", 1)[1]:
                        skipped += 1
                        continue
                    fh.write(line)
                    kept += 1
    tmp_out.rename(out)
    shutil.rmtree(parts)
    return kept, skipped


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
    ap.add_argument("--workers", type=int, default=16,
                    help="emilia: parallel directory scans (resumable); 1 = serial")
    args = ap.parse_args()

    if args.layout == "emilia" and args.workers > 1:
        if args.root is None:
            ap.error("emilia layout requires --root")
        kept, skipped = scan_emilia_parallel(args.root, args.out, args.workers,
                                             args.require_text)
        print(f"Wrote {kept} utterances to {args.out}"
              + (f" ({skipped} skipped: no transcript)" if skipped else ""))
        return

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
