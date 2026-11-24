"""Voice conversion with FaCT's streams (scripts/eval_vc.py).

Live speaker change keeps the source's content and prosody tokens and swaps
the speaker vector for a target speaker's. The speaker vector carries no
pitch (docs/CLAIMS.md section 3, 0.07 st), so a target with a different
pitch range is only reached by also moving the prosody stream:

  recon      own speaker vector (another utterance of the source speaker)
  vc         target speaker vector; content and prosody from the source
  vcpm       "pitch-matched": as vc, but the prosody tokens come from the
             source shifted by the speakers' median-F0 ratio (WORLD), the
             content (and residual) tokens from the unshifted source

Each source utterance is converted to two targets: a speaker in the same
pitch group and one in the other. Pitch group = speaker median F0 below or
above a threshold (default 160 Hz): a proxy for the usual male / female
split, since the eval sets carry no gender labels.

Library + merge (shards are merged with fact.eval.factorization merge).
"""

from __future__ import annotations

import math

import numpy as np

MIN_VOICED = 10


def speaker_median_f0(f0_tracks: list[np.ndarray]) -> float:
    """Median F0 (Hz) over the voiced frames of several utterances."""
    v = np.concatenate([f[f > 0] for f in f0_tracks]) if f0_tracks else np.array([])
    return float(np.median(v)) if len(v) >= MIN_VOICED else float("nan")


def pitch_groups(spk_f0: dict[str, float], threshold_hz: float) -> dict[str, str]:
    """{speaker: 'low' | 'high'}; speakers without a usable F0 are left out."""
    return {s: ("low" if f < threshold_hz else "high")
            for s, f in spk_f0.items() if math.isfinite(f)}


def choose_targets(groups: dict[str, str]) -> dict[str, dict[str, str | None]]:
    """For each speaker, a target in the same pitch group and one in the
    other group, deterministic and independent of input order: the i-th
    speaker of its sorted group takes the next speaker of its own group
    (cyclic) and the i-th (mod size) of the other group. None when a group
    has no other member."""
    by = {g: sorted(s for s, gg in groups.items() if gg == g) for g in ("low", "high")}
    out = {}
    for g, members in by.items():
        other = by["high" if g == "low" else "low"]
        for i, s in enumerate(members):
            same = members[(i + 1) % len(members)] if len(members) > 1 else None
            cross = other[i % len(other)] if other else None
            out[s] = {"same": same, "cross": cross}
    return out


def expected_shift_st(f0_target: float, f0_source: float, max_abs: float) -> float:
    """Semitones that move the source's median pitch onto the target's, clipped."""
    st = 12.0 * math.log2(f0_target / f0_source)
    return float(max(-max_abs, min(max_abs, st)))


def world_shift(wav: np.ndarray, sr: int, semitones: float, f0_min: float = 50.0,
                f0_max: float = 550.0) -> np.ndarray:
    """WORLD resynthesis with F0 scaled by 2^(semitones/12), same length as wav."""
    import pyworld as pw
    x = wav.astype(np.float64)
    f0, t = pw.dio(x, sr, f0_floor=f0_min, f0_ceil=f0_max, frame_period=5.0)
    f0 = pw.stonemask(x, f0, t, sr)
    if (f0 > 0).sum() < MIN_VOICED:
        raise ValueError("utterance has too few voiced frames for F0 manipulation")
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)
    y = pw.synthesize(np.ascontiguousarray(f0 * 2.0 ** (semitones / 12.0)), sp, ap, sr, 5.0)
    y = y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
    return y.astype(np.float32)


def contour_corr(f0_out: np.ndarray, f0_ref: np.ndarray) -> float:
    """Pearson correlation of log-F0 over frames voiced in both: is the
    source's intonation kept (whatever the absolute pitch)?"""
    n = min(len(f0_out), len(f0_ref))
    a, b = f0_out[:n], f0_ref[:n]
    m = (a > 0) & (b > 0)
    if m.sum() < MIN_VOICED:
        return float("nan")
    la, lb = np.log(a[m]), np.log(b[m])
    if la.std() == 0 or lb.std() == 0:
        return float("nan")
    return float(np.corrcoef(la, lb)[0, 1])
