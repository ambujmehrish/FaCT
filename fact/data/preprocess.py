"""Offline preprocessing: manifest (wav + transcript) -> training shards.

CPU-bound; run once, store shards. Parallel across worker processes and
resumable: the manifest is split into fixed groups, each group maps to one
shard file, and existing shard files are skipped - re-run the same command
after an interruption and it continues where it left off.

Each shard is a .pt file holding a list of dicts:
  mel:  (T_mel, n_mels) float16 log-mel
  f0:   (T_mel,) float32 F0 in Hz (0 = unvoiced; pyworld, or ACF fallback)
  text: (S,) int64 byte-level label ids (0 reserved for CTC blank)

After sharding, dataset statistics (global mel mean/std, hours) are written
to <out>/stats.json - set audio.mel_mean/mel_std in your config from these
(flow matching wants ~unit-scale targets).

Usage:
    python -m fact.data.preprocess --manifest train.tsv --out shards/ \
        --config configs/fact_base.yaml --workers 16
    python -m fact.data.preprocess --out shards/ --stats-only
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from ..config import FaCTConfig
from ..modules.mel import LogMelSpectrogram
from .prosody_targets import extract_f0

# Per-worker globals (initialized once per process).
_CFG: FaCTConfig | None = None
_MEL: LogMelSpectrogram | None = None


def encode_text_bytes(text: str, vocab_size: int) -> torch.Tensor:
    """Byte-level ids shifted by 1 (0 is the CTC blank), clamped to vocab."""
    ids = torch.tensor(list(text.encode("utf-8")), dtype=torch.long) + 1
    return ids.clamp(1, vocab_size - 1)


def load_audio(path: str, sample_rate: int) -> np.ndarray:
    """Load and resample to mono float32.

    soundfile first, torchaudio second (some codecs only one can read);
    if both fail the error carries BOTH causes - never a silent skip.
    """
    errors: list[str] = []
    wav = sr = None
    try:
        import soundfile as sf
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as e:
        errors.append(f"soundfile: {type(e).__name__}: {e}")
    if wav is None:
        try:
            import torchaudio
            t, sr = torchaudio.load(path)
            wav = t.mean(0).numpy()
        except Exception as e:
            errors.append(f"torchaudio: {type(e).__name__}: {e}")
            raise RuntimeError(f"Cannot decode {path}: " + " | ".join(errors))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != sample_rate:
        import torchaudio  # resampling requires torchaudio; fail loudly if absent
        wav = torchaudio.functional.resample(
            torch.from_numpy(np.ascontiguousarray(wav)), sr, sample_rate
        ).numpy()
    return wav.astype(np.float32)


_F0_METHOD = "pyworld"


def _worker_init(cfg_dict: dict, min_sec: float, max_sec: float,
                 f0_method: str = "pyworld") -> None:
    global _CFG, _MEL, _F0_METHOD
    torch.set_num_threads(1)
    _CFG = FaCTConfig.from_dict(cfg_dict)
    _MEL = LogMelSpectrogram(_CFG.audio)
    _F0_METHOD = f0_method
    _set_filters(min_sec, max_sec)


def _process_one(line: str) -> dict | str:
    """One manifest line -> shard item, or an error string (kept for reporting)."""
    cfg, mel_fn = _CFG, _MEL
    try:
        parts = line.split("\t", 1)
        wav_path = parts[0]
        transcript = parts[1] if len(parts) > 1 else ""
        wav = load_audio(wav_path, cfg.audio.sample_rate)
        seconds = len(wav) / cfg.audio.sample_rate
        if not (cfg_min_sec <= seconds <= cfg_max_sec):
            return f"filtered({seconds:.1f}s): {wav_path}"
        with torch.no_grad():
            mel = mel_fn(torch.from_numpy(wav).unsqueeze(0)).squeeze(0)
        f0 = extract_f0(wav, cfg.audio.sample_rate, cfg.audio.hop_length,
                        cfg.prosody.f0_min, cfg.prosody.f0_max, method=_F0_METHOD)
        t = min(mel.shape[0], len(f0))
        return {
            "mel": mel[:t].to(torch.float16),
            "f0": torch.from_numpy(np.ascontiguousarray(f0[:t])).float(),
            "text": encode_text_bytes(transcript, cfg.ctc.vocab_size),
        }
    except Exception as e:  # keep going; one bad file must not kill a 60K-h run
        return f"error({type(e).__name__}: {e}): {line[:120]}"


# Filter bounds live at module level so workers see them after fork/spawn init.
cfg_min_sec = 1.0
cfg_max_sec = 30.0


def _set_filters(min_sec: float, max_sec: float) -> None:
    global cfg_min_sec, cfg_max_sec
    cfg_min_sec, cfg_max_sec = min_sec, max_sec


def config_fingerprint(cfg: FaCTConfig) -> dict:
    """The config fields baked into shard contents; a mismatch means shards
    must be rebuilt, not reused."""
    return {
        "sample_rate": cfg.audio.sample_rate,
        "hop_length": cfg.audio.hop_length,
        "n_mels": cfg.audio.n_mels,
        "text_vocab_size": cfg.ctc.vocab_size,
    }


def compute_stats(out_dir: Path, hop_length: int, sample_rate: int,
                  fingerprint: dict | None = None) -> dict:
    """Global mel mean/std + corpus size, from the written shards.

    fingerprint: stamped into stats.json by preprocessing runs; when None
    (--stats-only), any existing fingerprint is preserved - recomputing
    stats must not forge a fresh-preprocess claim over old shards.
    """
    total = 0.0
    total_sq = 0.0
    n_vals = 0
    n_frames = 0
    n_utts = 0
    for shard_path in sorted(out_dir.glob("shard_*.pt")):
        for item in torch.load(shard_path, map_location="cpu", weights_only=True):
            mel = item["mel"].float()
            total += mel.sum().item()
            total_sq += mel.pow(2).sum().item()
            n_vals += mel.numel()
            n_frames += mel.shape[0]
            n_utts += 1
    if n_vals == 0:
        return {"n_utterances": 0, "hours": 0.0}
    mean = total / n_vals
    std = max((total_sq / n_vals - mean ** 2), 1e-8) ** 0.5
    stats = {
        "n_utterances": n_utts,
        "hours": n_frames * hop_length / sample_rate / 3600.0,
        "mel_mean": round(mean, 4),
        "mel_std": round(std, 4),
    }
    if fingerprint is None:
        p = out_dir / "stats.json"
        if p.exists():
            fingerprint = json.loads(p.read_text()).get("fingerprint")
    if fingerprint is not None:
        stats["fingerprint"] = fingerprint
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def preprocess(manifest: str, out_dir: str, cfg: FaCTConfig,
               shard_size: int = 512, workers: int = 1,
               min_sec: float = 1.0, max_sec: float = 30.0,
               f0_method: str = "pyworld",
               max_drop_rate: float = 0.05) -> dict:
    # Validate the F0 method BEFORE spawning workers: a missing pyworld must
    # abort the run, not fail per-utterance until the drop guard trips.
    if f0_method == "pyworld":
        try:
            import pyworld  # noqa: F401
        except ImportError:
            raise SystemExit(
                "pyworld is not installed (pip install -e '.[audio]'). "
                "Refusing to run - pass --f0 autocorr only if you knowingly "
                "want the coarse fallback extractor."
            )
    elif f0_method != "autocorr":
        raise SystemExit(f"Unknown --f0 method {f0_method!r}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = config_fingerprint(cfg)
    # Resume-mixing guard: never append shards with a new encoding to a
    # directory built under a different config.
    stats_path = out / "stats.json"
    if stats_path.exists():
        old_fp = json.loads(stats_path.read_text()).get("fingerprint")
        if old_fp is not None and old_fp != fingerprint:
            raise SystemExit(
                f"{out} already contains shards preprocessed with "
                f"{old_fp}, but the current config implies {fingerprint}. "
                f"Use a fresh --out directory (mixing encodings corrupts "
                f"training data silently)."
            )
    drops_log = out / "drops.log"
    drops_log.unlink(missing_ok=True)  # this run's drops only

    with open(manifest, encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    groups = [lines[i : i + shard_size] for i in range(0, len(lines), shard_size)]
    print(f"{len(lines)} utterances -> {len(groups)} shards "
          f"(size {shard_size}, {workers} workers, f0={f0_method})")

    n_done = n_skipped_shards = n_filtered = n_errors = 0
    pool = Pool(workers, initializer=_worker_init,
                initargs=(cfg.to_dict(), min_sec, max_sec, f0_method)) \
        if workers > 1 else None
    if pool is None:
        _worker_init(cfg.to_dict(), min_sec, max_sec, f0_method)
    try:
        for gi, group in enumerate(groups):
            shard_path = out / f"shard_{gi:06d}.pt"
            if shard_path.exists():
                n_skipped_shards += 1
                continue
            results = pool.map(_process_one, group) if pool else \
                [_process_one(ln) for ln in group]
            items = [r for r in results if isinstance(r, dict)]
            dropped = [r for r in results if isinstance(r, str)]
            # Duration filtering is intentional; errors are not - only the
            # latter count toward the abort threshold.
            errors = [d for d in dropped if d.startswith("error")]
            n_errors += len(errors)
            n_filtered += len(dropped) - len(errors)
            if dropped:
                with open(drops_log, "a", encoding="utf-8") as fh:
                    fh.writelines(f"shard {gi}: {msg}\n" for msg in dropped)
                for msg in errors[:3]:
                    print(f"  drop: {msg}")
            if items:
                tmp = shard_path.with_suffix(".pt.tmp")
                torch.save(items, tmp)
                tmp.rename(shard_path)  # atomic: resume never sees partial shards
            n_done += len(items)
            print(f"shard {gi + 1}/{len(groups)}: {len(items)} items "
                  f"({n_filtered} filtered, {n_errors} errors so far)")
    finally:
        if pool:
            pool.close()
            pool.join()

    if n_skipped_shards:
        print(f"Resumed: skipped {n_skipped_shards} existing shards")
    processed = n_done + n_filtered + n_errors
    if processed and n_errors / processed > max_drop_rate:
        raise SystemExit(
            f"ABORT: {n_errors}/{processed} utterances "
            f"({100 * n_errors / processed:.1f}%) failed with errors - above "
            f"the --max-drop-rate threshold ({100 * max_drop_rate:.0f}%). "
            f"This usually means a systematic problem (bad paths, codec, "
            f"sample rate), not bad data. See {drops_log}. Shards written so "
            f"far are kept; fix the cause and re-run to fill the gaps."
        )
    if n_filtered:
        print(f"{n_filtered} utterances filtered by duration bounds "
              f"({min_sec}-{max_sec}s) - logged in {drops_log}")
    stats = compute_stats(out, cfg.audio.hop_length, cfg.audio.sample_rate,
                          fingerprint=fingerprint)
    print(f"Stats: {stats}")
    if "mel_mean" in stats:
        print(f"-> set audio.mel_mean: {stats['mel_mean']} and "
              f"audio.mel_std: {stats['mel_std']} in your config")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--shard-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--min-sec", type=float, default=1.0)
    ap.add_argument("--max-sec", type=float, default=30.0)
    ap.add_argument("--f0", default="pyworld", choices=["pyworld", "autocorr"],
                    help="F0 extractor; autocorr is a coarse explicit opt-in")
    ap.add_argument("--max-drop-rate", type=float, default=0.05,
                    help="abort if more than this fraction of utterances drop")
    ap.add_argument("--stats-only", action="store_true",
                    help="recompute stats.json from existing shards and exit")
    args = ap.parse_args()
    cfg = FaCTConfig.from_yaml(args.config) if args.config else FaCTConfig()
    if args.stats_only:
        print(compute_stats(Path(args.out), cfg.audio.hop_length, cfg.audio.sample_rate))
        return
    if not args.manifest:
        ap.error("--manifest is required unless --stats-only")
    preprocess(args.manifest, args.out, cfg, args.shard_size, args.workers,
               args.min_sec, args.max_sec, args.f0, args.max_drop_rate)


if __name__ == "__main__":
    main()
