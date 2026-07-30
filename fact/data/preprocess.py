"""Offline preprocessing: wav + transcript -> training shards.

CPU-bound; run once and store shards (mel + F0 + byte-level text). Requires
the `audio` extra (torchaudio/soundfile + pyworld).

Usage:
    python -m fact.data.preprocess \
        --manifest manifest.tsv --out shards/ --config configs/fact_base.yaml

Manifest format (TSV): <wav_path>\t<transcript>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..config import FaCTConfig
from ..modules.mel import LogMelSpectrogram
from .prosody_targets import extract_f0_pyworld


def encode_text_bytes(text: str, vocab_size: int) -> torch.Tensor:
    """Byte-level ids shifted by 1 (0 is the CTC blank), clamped to vocab."""
    ids = torch.tensor(list(text.encode("utf-8")), dtype=torch.long) + 1
    return ids.clamp(1, vocab_size - 1)


def load_wav(path: str, sample_rate: int) -> np.ndarray:
    try:
        import soundfile as sf
        wav, sr = sf.read(path, dtype="float32")
    except ImportError:
        import torchaudio
        t, sr = torchaudio.load(path)
        wav = t.mean(0).numpy()
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != sample_rate:
        import torchaudio
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav), sr, sample_rate
        ).numpy()
    return wav


def preprocess(manifest: str, out_dir: str, cfg: FaCTConfig,
               shard_size: int = 512) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mel_fn = LogMelSpectrogram(cfg.audio)

    shard: list[dict] = []
    shard_id = 0

    def flush():
        nonlocal shard, shard_id
        if shard:
            torch.save(shard, out / f"shard_{shard_id:06d}.pt")
            shard_id += 1
            shard = []

    with open(manifest) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            wav_path, transcript = line.split("\t", 1)
            wav = load_wav(wav_path, cfg.audio.sample_rate)
            with torch.no_grad():
                mel = mel_fn(torch.from_numpy(wav).unsqueeze(0)).squeeze(0)
            f0 = extract_f0_pyworld(
                wav, cfg.audio.sample_rate, cfg.audio.hop_length,
                cfg.prosody.f0_min, cfg.prosody.f0_max,
            )
            t = min(mel.shape[0], len(f0))
            shard.append({
                "mel": mel[:t].to(torch.float16),
                "f0": torch.from_numpy(f0[:t]).float(),
                "text": encode_text_bytes(transcript, cfg.ctc.vocab_size),
            })
            if len(shard) >= shard_size:
                flush()
    flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--shard-size", type=int, default=512)
    args = ap.parse_args()
    cfg = FaCTConfig.from_yaml(args.config) if args.config else FaCTConfig()
    preprocess(args.manifest, args.out, cfg, args.shard_size)


if __name__ == "__main__":
    main()
