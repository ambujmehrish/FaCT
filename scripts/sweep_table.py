"""Dev-set inference sweep table (scripts/submit_dev_sweep.sh outputs).

    python scripts/sweep_table.py --out results/tables/dev_sweep.md

Per run and setting: band mel-L1 (mean, 95% bootstrap CI) for the full and
tokens-only views, split into speech-active and silent frames, and - for
the 8-sample job - the mel-L1 of the mean of 8 samples. The selection rule
(fixed in submit_dev_sweep.sh before any result): lowest full-view mel-L1;
marked with *.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "mrt", Path(__file__).resolve().parent / "make_recon_table.py")
_mrt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mrt)


def view_stats(r: dict, view: str, rng) -> dict:
    per = r["per_utterance"].values()
    out = {}
    keys = {"mel_l1_band": "mel_l1_band", "mel_l1_band_active": "mel_l1_band_active",
            "mel_l1_band_silent": "mel_l1_band_silent",
            f"mel_l1_band_mean{r.get('n_samples', 1)}": "meanK"}
    for key, name in keys.items():
        vals = np.array([m[f"{view}/{key}"] for m in per if f"{view}/{key}" in m])
        if len(vals):
            out[name] = _mrt.boot_mean(vals, rng)
    return out


def fmt(t) -> str:
    return "-" if t is None else f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/fact_recon/sweep/*.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for f in sorted(glob.glob(args.glob)):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        run = Path(f).stem.split("_T")[0]
        views = list(r["views"])
        tok = next((v for v in views if v != "full"), None)
        rows.append({"run": run, "T": r.get("temperature", 1.0), "N": r["nfe"],
                     "G": r.get("cfg_scale", 1.0), "K": r.get("n_samples", 1),
                     "n": r["n_utterances"], "full": view_stats(r, "full", rng),
                     "tok_view": tok, "tok": view_stats(r, tok, rng) if tok else {}})
    lines = ["## Inference sweep - Emilia dev (model-selection set)", "",
             "Band mel-L1 ↓, mean [95% bootstrap CI]. active = frames within 40 dB of the "
             "loudest frame; mean-of-K = L1 of the average of K samples (K=8 job only). "
             "* = selected (lowest full-view mel-L1 per run, rule fixed in advance).", "",
             "| run | temp | NFE | CFG | K | n | full | full active | full silent | full mean-of-K | "
             "tokens view | tokens | tokens active | tokens silent |",
             "|" + "---|" * 14]
    for run in sorted({r["run"] for r in rows}):
        rs = sorted((r for r in rows if r["run"] == run), key=lambda r: (r["K"], -r["T"], r["N"], r["G"]))
        single = [r for r in rs if r["K"] == 1 and "mel_l1_band" in r["full"]]
        best = min(single, key=lambda r: r["full"]["mel_l1_band"][0]) if single else None
        for r in rs:
            f, t = r["full"], r["tok"]
            mark = "*" if r is best else ""
            lines.append(f"| {run}{mark} | {r['T']} | {r['N']} | {r['G']} | {r['K']} | {r['n']} | "
                         f"{fmt(f.get('mel_l1_band'))} | {fmt(f.get('mel_l1_band_active'))} | "
                         f"{fmt(f.get('mel_l1_band_silent'))} | {fmt(f.get('meanK'))} | "
                         f"{r['tok_view'] or '-'} | {fmt(t.get('mel_l1_band'))} | "
                         f"{fmt(t.get('mel_l1_band_active'))} | {fmt(t.get('mel_l1_band_silent'))} |")
    text = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
