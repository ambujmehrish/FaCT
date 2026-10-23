"""Reconstruction table: baselines + FaCT views on one eval set, with CIs.

    python scripts/make_recon_table.py \
        --baselines results/baseline_recon_ls/*.json \
        --fact fact_main=results/fact_recon/fact_main_ls.json \
        --title "LibriSpeech test-clean" --out results/tables/recon_ls.md

Every number is recomputed from per-utterance values:
  - mean with a bootstrap 95% CI over utterances (2,000 resamples, fixed seed);
  - WER is corpus-level (total errors / total reference words), with the CI
    from resampling utterances;
  - "vs FaCT" columns are PAIRED: per-utterance difference on the utterances
    both systems scored, so utterance difficulty cancels.
FaCT rows are pre-vocoder (mel domain) until the stage-C vocoder exists:
waveform metrics (PESQ/STOI/WER/F0) are shown as "n/a" for them, never
filled from elsewhere.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_BOOT = 2000
COLS = [  # (key, header, higher_is_better, fmt)
    ("pesq_wb", "PESQ-WB ↑", True, "{:.2f}"),
    ("stoi", "STOI ↑", True, "{:.3f}"),
    ("wer", "WER % ↓", False, "{:.2f}"),
    ("mel_l1_band", "mel-L1 ↓", False, "{:.3f}"),
    ("f0_rmse_hz", "F0 RMSE ↓", False, "{:.1f}"),
    ("voicing_f1", "V/UV F1 ↑", True, "{:.3f}"),
    ("energy_corr", "energy r ↑", True, "{:.3f}"),
    ("utmos", "UTMOS ↑", True, "{:.2f}"),
    ("spk_sim", "SIM ↑", True, "{:.3f}"),
]


CHUNK = 100  # resamples per batch: keeps peak memory small (login nodes)


def _resample_idx(n: int, rng):
    for _ in range(0, N_BOOT, CHUNK):
        yield rng.integers(0, n, size=(CHUNK, n), dtype=np.int32)


def boot_mean(x: np.ndarray, rng) -> tuple[float, float, float]:
    m = np.concatenate([x[idx].mean(axis=1) for idx in _resample_idx(len(x), rng)])
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_ratio(err: np.ndarray, words: np.ndarray, rng) -> tuple[float, float, float]:
    r = np.concatenate([err[idx].sum(axis=1) / np.maximum(words[idx].sum(axis=1), 1)
                        for idx in _resample_idx(len(err), rng)])
    return (float(err.sum() / max(words.sum(), 1)),
            float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)))


def load_baseline(path: Path) -> dict:
    r = json.loads(path.read_text(encoding="utf-8"))
    per = {}
    for utt, m in r["per_utterance"].items():
        row = {k: v for k, v in m.items() if k != "extras" and isinstance(v, (int, float))}
        row.update({k: v for k, v in (m.get("extras") or {}).items()})
        per[utt] = row
    bps = r["metadata"].get("discrete_bps")
    return {"name": r["baseline"], "kbps": bps / 1000 if bps else None, "per": per,
            "fact": False}


def load_fact(label: str, path: Path) -> list[dict]:
    r = json.loads(path.read_text(encoding="utf-8"))
    br = r["bitrate"]
    disc = br.get("discrete_bps", 0.0)
    res = br.get("residual_bps")
    # Tokens-only mode the model was trained for: residual-dropout models saw
    # "no residual" in training (discrete_only); others only the channel prior.
    dropout = (r.get("config", {}).get("bottleneck", {}) or {}).get("residual_dropout", 0.0)
    native = "discrete_only" if dropout and "discrete_only" in r["views"] else "tokens_prior"
    rows = []
    for view in r["views"]:
        per = {utt: {k.split("/", 1)[1]: v for k, v in m.items() if k.startswith(view + "/")}
               for utt, m in r["per_utterance"].items()}
        kbps = (disc + (res or 0.0)) / 1000 if view == "full" else disc / 1000
        note = {"full": "tokens + residual", "tokens_prior": "tokens only",
                "discrete_only": "tokens only (residual removed)"}.get(view, view)
        rows.append({"name": f"FaCT {label} ({note})", "kbps": kbps, "per": per,
                     "fact": True, "unbounded_residual": view == "full" and res is None,
                     "label": label, "view": view, "note": note,
                     "native_tokens": view == native})
    return rows


def cell(sys: dict, key: str, fmt: str, rng) -> str:
    per = sys["per"]
    if key == "wer":
        utts = [u for u, m in per.items() if "wer_errors" in m]
        if not utts:
            return "n/a" if sys["fact"] else "-"
        err = np.array([per[u]["wer_errors"] for u in utts])
        words = np.array([per[u]["wer_words"] for u in utts])
        m, lo, hi = (100 * v for v in boot_ratio(err, words, rng))
    else:
        vals = np.array([m[key] for m in per.values() if m.get(key) is not None
                         and np.isfinite(m[key])])
        if len(vals) == 0:
            return "n/a" if sys["fact"] else "-"
        m, lo, hi = boot_mean(vals, rng)
    return f"{fmt.format(m)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def paired(a: dict, b: dict, key: str, rng) -> str:
    shared = [u for u in a["per"] if u in b["per"] and a["per"][u].get(key) is not None
              and b["per"][u].get(key) is not None]
    if len(shared) < 10:
        return "-"
    d = np.array([a["per"][u][key] - b["per"][u][key] for u in shared])
    m, lo, hi = boot_mean(d, rng)
    sig = "*" if lo > 0 or hi < 0 else ""
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", nargs="+", type=Path, default=[])
    ap.add_argument("--fact", nargs="*", default=[], metavar="LABEL=JSON")
    ap.add_argument("--title", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ref", default=None,
                    help="FaCT label for the paired-difference table (default: first --fact)")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    systems = [load_baseline(p) for p in args.baselines]
    facts = []
    for spec in args.fact:
        label, _, path = spec.partition("=")
        facts += load_fact(label, Path(path))
    systems = sorted(systems, key=lambda s: s["kbps"] or 0) + facts
    # Every system on the SAME utterances: when some rows cover a subset
    # (e.g. the other-utterance speaker-reference protocol keeps only
    # speakers with two utterances), all rows are restricted to the common set.
    common = set.intersection(*(set(s["per"]) for s in systems)) if systems else set()
    restricted = any(len(s["per"]) != len(common) for s in systems)
    if restricted:
        if not common:
            raise SystemExit("no utterance is shared by all systems")
        for s_ in systems:
            s_["per"] = {u: m for u, m in s_["per"].items() if u in common}
    n = {len(s["per"]) for s in systems}

    lines = [f"## {args.title}", "",
             f"Mean [95% bootstrap CI] over utterances (n = {', '.join(map(str, sorted(n)))}"
             f"{'; every row restricted to the utterances all systems share' if restricted else ''}). "
             "WER is corpus-level vs the ground-truth transcript. mel-L1 uses bins below the "
             "source audio's Nyquist. FaCT rows are pre-vocoder (mel domain): waveform metrics "
             "are n/a until the stage-C vocoder exists.", "",
             "| system | kbps | " + " | ".join(h for _, h, _, _ in COLS) + " |",
             "|" + "---|" * (len(COLS) + 2)]
    for s in systems:
        kb = "-" if s["kbps"] is None else f"{s['kbps']:.3f}"
        if s.get("unbounded_residual"):
            kb += " + unbounded residual"
        lines.append(f"| {s['name']} | {kb} | " +
                     " | ".join(cell(s, k, f, rng) for k, _, _, f in COLS) + " |")

    ref_label = args.ref or (facts[0]["label"] if facts else None)
    refs = [f for f in facts if f["label"] == ref_label and (f["view"] == "full" or f["native_tokens"])]
    if args.ref and not refs:
        raise SystemExit(f"--ref {args.ref!r} is not one of the --fact labels")
    if refs and len(systems) > len(facts):
        lines += ["", f"**Paired mel-L1 difference, FaCT {ref_label} minus each baseline** "
                  "(per utterance, same audio; negative = FaCT better; * = 95% CI excludes 0). "
                  "Tokens-only uses the mode the model was trained for (residual removed for "
                  "residual-dropout models, channel prior otherwise).", "",
                  "| baseline | kbps | " + " | ".join(
                      f"Δ vs {r['note']} ({r['kbps']:.3f} kbps)"
                      for r in refs) + " |",
                  "|" + "---|" * (len(refs) + 2)]
        for s in systems:
            if s["fact"]:
                continue
            kb = "-" if s["kbps"] is None else f"{s['kbps']:.3f}"
            lines.append(f"| {s['name']} | {kb} | " +
                         " | ".join(paired(r, s, "mel_l1_band", rng) for r in refs) + " |")

    text = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
