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
