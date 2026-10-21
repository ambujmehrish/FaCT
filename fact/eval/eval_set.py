"""Eval-set loading shared by the baseline and FaCT reconstruction evals."""

from __future__ import annotations

from pathlib import Path

import torch


def load_eval_wavs(eval_dir: Path, limit: int | None) -> list[tuple[str, torch.Tensor, int]]:
    import soundfile as sf
    paths = sorted(p for p in eval_dir.rglob("*") if p.suffix.lower() in (".wav", ".flac"))
    if limit:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No .wav/.flac under {eval_dir}")
    out = []
    for p in paths:
        wav, sr = sf.read(p, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        out.append((str(p.relative_to(eval_dir)), torch.from_numpy(wav), sr))
    return out


def load_eval_manifest(manifest: Path, limit: int | None,
                       sample_rate: int) -> list[tuple[str, torch.Tensor, int]]:
    """path<TAB>text manifest (e.g. Emilia test.tsv, mp3) -> eval tuples.

    Audio is decoded with the preprocessing loader at `sample_rate` (Emilia
    is natively 24 kHz). The key is the file stem, unique within Emilia.
    """
    from ..data.preprocess import load_audio
    out = []
    with open(manifest, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            path = line.split("\t", 1)[0]
            out.append((Path(path).stem, torch.from_numpy(load_audio(path, sample_rate)),
                        sample_rate))
            if limit and len(out) >= limit:
                break
    if not out:
        raise FileNotFoundError(f"no entries in {manifest}")
    return out


def load_transcripts(eval_dir: Path | None = None,
                     manifest: Path | None = None) -> dict[str, str]:
    """Ground-truth text keyed like the eval tuples: <rel>.txt next to each
    wav (prepare_eval_set.py), or the manifest's text column."""
    texts: dict[str, str] = {}
    if manifest is not None:
        with open(manifest, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    path, _, text = line.rstrip("\n").partition("\t")
                    texts[Path(path).stem] = text
    elif eval_dir is not None:
        for p in sorted(eval_dir.rglob("*.txt")):
            for ext in (".wav", ".flac"):
                if p.with_suffix(ext).exists():
                    texts[str(p.with_suffix(ext).relative_to(eval_dir))] = \
                        p.read_text(encoding="utf-8").strip()
    return texts
