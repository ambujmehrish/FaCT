"""Factorization evaluation: do FaCT's streams carry what they claim?

Controlled manipulations (WORLD analysis/resynthesis, all variants of an
utterance pass through WORLD so its artifacts cancel):
  w0      resynthesis, unchanged F0
  s+K/-K  F0 shifted by K semitones (same timing, same spectral envelope)
  flat    F0 replaced by the utterance's median (monotone)

Swaps are decoded with the stage-D waveform decoder (deterministic), the
condition being content embedding of one variant + prosody embedding of
another (+ residual of the content source in the "full" view):
  recon          content w0, prosody w0          (reference output)
  prosody<s>     content w0, prosody s           expect pitch shift = s
  content<s>     content s,  prosody w0          expect pitch shift = 0
  both<s>        content s,  prosody s           expect s (sanity)
  prosody_flat   content w0, prosody flat        expect intonation ratio << 1
  vc             tokens w0, speaker reference from ANOTHER speaker
Pitch shifts are measured against the `recon` output (so the codec's own
pitch error cancels): median 12*log2(F0_out / F0_recon) over frames voiced
in both. Also: content / prosody token agreement between w0 and each
variant (a factorized content stream should barely change under a pure
pitch change), Whisper WER per condition, and speaker similarity of `vc`
to the target vs the source speaker.

Library + merge CLI (shards written by scripts/eval_factorization.py):
    python -m fact.eval.factorization merge --inputs 'results/factorization/x_shard*.json' \
        --out results/factorization/x.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

import numpy as np

WORLD_FRAME_MS = 5.0
MIN_VOICED = 10  # frames voiced in both tracks needed for a pitch statistic


# ------------------------------------------------------------- manipulation

def world_variants(wav: np.ndarray, sr: int, shifts: list[int], f0_min: float = 50.0,
                   f0_max: float = 550.0) -> dict[str, np.ndarray]:
    """{'w0', 's+K', 's-K', 'flat'} resyntheses, each exactly len(wav) samples."""
    import pyworld as pw
    x = wav.astype(np.float64)
    f0, t = pw.dio(x, sr, f0_floor=f0_min, f0_ceil=f0_max, frame_period=WORLD_FRAME_MS)
    f0 = pw.stonemask(x, f0, t, sr)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)

    def synth(f):
        y = pw.synthesize(np.ascontiguousarray(f), sp, ap, sr, WORLD_FRAME_MS)
        y = y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
        return y.astype(np.float32)

    voiced = f0 > 0
    if voiced.sum() < MIN_VOICED:
        raise ValueError("utterance has too few voiced frames for F0 manipulation")
    out = {"w0": synth(f0)}
    for s in shifts:
        out[f"s{s:+d}"] = synth(f0 * 2.0 ** (s / 12.0))
    out["flat"] = synth(np.where(voiced, np.median(f0[voiced]), 0.0))
    return out


# ------------------------------------------------------------------ metrics

def pitch_shift_st(f0_out: np.ndarray, f0_ref: np.ndarray) -> float:
    """Median pitch difference in semitones over frames voiced in both."""
    n = min(len(f0_out), len(f0_ref))
    a, b = f0_out[:n], f0_ref[:n]
    m = (a > 0) & (b > 0)
    if m.sum() < MIN_VOICED:
        return float("nan")
    return float(np.median(12.0 * np.log2(a[m] / b[m])))


def intonation_ratio(f0_out: np.ndarray, f0_ref: np.ndarray) -> float:
    """Spread of log-F0 of the output over that of the reference (each on its
    own voiced frames). Spread = median absolute deviation, so a few octave
    errors or voicing-edge frames cannot dominate it."""
    def sd(f):
        v = np.log(f[f > 0])
        return float(np.median(np.abs(v - np.median(v)))) if len(v) >= MIN_VOICED else float("nan")
    r = sd(f0_ref)
    return sd(f0_out) / r if r and math.isfinite(r) and r > 0 else float("nan")


def token_agreement(a, b) -> float:
    """Fraction of identical indices (trimmed to the common length)."""
    import torch
    a, b = torch.as_tensor(a), torch.as_tensor(b)
    n = min(a.shape[-1], b.shape[-1])
    return float((a[..., :n] == b[..., :n]).float().mean())


def median_f0(f0: np.ndarray) -> float:
    v = f0[f0 > 0]
    return float(np.median(v)) if len(v) >= MIN_VOICED else float("nan")


# -------------------------------------------------------------------- merge

def _boot(x: np.ndarray, rng, n: int = 2000) -> tuple[float, float, float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def aggregate(per_utt: dict[str, dict]) -> dict:
    """{metric: {mean, lo, hi, n}} over utterances; WER is corpus-level."""
    rng = np.random.default_rng(0)
    keys = sorted({k for m in per_utt.values() for k in m})
    out = {}
    for k in keys:
        if k.endswith("/wer_errors") or k.endswith("/wer_words"):
            continue
        vals = np.array([m[k] for m in per_utt.values() if k in m and m[k] is not None], dtype=float)
        mean, lo, hi = _boot(vals, rng)
        out[k] = {"mean": mean, "lo": lo, "hi": hi, "n": int(np.isfinite(vals).sum())}
    for k in keys:
        if k.endswith("/wer_errors"):
            base = k[: -len("/wer_errors")]
            e = sum(m[k] for m in per_utt.values() if k in m)
            w = sum(m.get(base + "/wer_words", 0) for m in per_utt.values() if k in m)
            out[base + "/wer"] = {"mean": 100.0 * e / max(w, 1), "n": sum(k in m for m in per_utt.values())}
    return out


def merge(inputs: list[str], out: Path) -> dict:
    shards = [json.loads(Path(f).read_text(encoding="utf-8")) for f in sorted(inputs)]
    if not shards:
        raise SystemExit("no shard files")
    meta = {k: v for k, v in shards[0].items() if k not in ("per_utterance", "shard", "skipped")}
    for s in shards[1:]:
        diff = [k for k in meta if s.get(k) != meta[k]]
        if diff:
            raise SystemExit(f"shards disagree on {diff}")
    shard_ids = sorted(s["shard"] for s in shards)
    if shard_ids != list(range(meta["num_shards"])):
        raise SystemExit(f"shards {shard_ids} present, expected 0..{meta['num_shards'] - 1}")
    per, skipped = {}, {}
    for s in shards:
        skipped.update(s.get("skipped", {}))
        dup = set(per) & set(s["per_utterance"])
        if dup:
            raise SystemExit(f"utterances in two shards: {sorted(dup)[:3]}")
        per.update(s["per_utterance"])
    res = meta | {"n_utterances": len(per), "n_shards": len(shards), "skipped": skipped,
                  "aggregate": aggregate(per), "per_utterance": per}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.{os.getpid()}.partial")
    tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    os.replace(tmp, out)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m fact.eval.factorization")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--inputs", required=True, help="glob of shard JSONs")
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--expect-shards", type=int, default=None)
    a = ap.parse_args()
    files = sorted(glob.glob(a.inputs))
    if a.expect_shards is not None and len(files) != a.expect_shards:
        raise SystemExit(f"{len(files)} shard files match {a.inputs}, expected {a.expect_shards}")
    res = merge(files, a.out)
    agg = res["aggregate"]
    for k in sorted(agg):
        v = agg[k]
        ci = f" [{v['lo']:.3f}, {v['hi']:.3f}]" if "lo" in v else ""
        print(f"{k:45s} {v['mean']:.3f}{ci} n={v['n']}")


if __name__ == "__main__":
    main()
