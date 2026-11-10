"""Enrolment ablation summary: FaCT stage-D rows decoded with the normal
enrolment (spk_ref=other), with no enrolment (spk_ref=mean: one fixed mean
speaker vector) and with a wrong speaker (spk_ref=xspk), on the same test
utterances. Paired 95% bootstrap CIs (scripts/make_recon_table.py).

    python scripts/summarize_enrolment.py --out results/enrolment_ablation/summary_paired.md

Also compares the no-enrolment rows with the baseline each FaCT model is
compared with in docs/CLAIMS.md (neither side gets enrolment then).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_recon_table", ROOT / "scripts/make_recon_table.py")
mrt = importlib.util.module_from_spec(_spec)
sys.modules["make_recon_table"] = mrt
_spec.loader.exec_module(mrt)

MODELS = {"disjref": ("factwave_main_c8_rd05_disjref@0250000", "focalcodec_25hz", "FocalCodec 25 Hz"),
          "la4": ("factwave_main_c8_rd05_la4@0250000", "focalcodec_50hz_2k_causal", "FocalCodec-Stream")}
SETS = {"ls": ("results/baseline_recon_ls", "LibriSpeech test-clean"),
        "emilia": ("results/baseline_recon_emilia", "Emilia test")}
KEYS = (("wer", "WER %"), ("utmos", "UTMOS"), ("spk_sim", "SIM"))


def _abs(s: dict, key: str) -> str:
    vals = [m[key] for m in s["per"].values() if m.get(key) is not None]
    if key == "wer":
        e = sum(m["wer_errors"] for m in s["per"].values() if "wer_errors" in m)
        w = sum(m["wer_words"] for m in s["per"].values() if "wer_errors" in m)
        return f"{100 * e / max(w, 1):.2f}"
    return f"{sum(vals) / len(vals):.3f}" if vals else "-"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    lines = ["# Enrolment ablation (test sets; paired 95% bootstrap CIs, * = CI excludes 0)", "",
             "- other: enrolment = another utterance of the same speaker (every reported FaCT row).",
             "- mean: one fixed speaker vector (the eval set's mean direction) for all: no enrolment.",
             "- xspk: enrolment from a different speaker.", ""]
    for sk, (sdir, sname) in SETS.items():
        for mk, (row, base, bname) in MODELS.items():
            for view in ("full", "discrete_only"):
                paths = {m: (ROOT / sdir / f"{row}_{view}_spkother.json") if m == "other"
                         else ROOT / "results/enrolment_ablation" / sk / f"{row}_{view}_spk{m}.json"
                         for m in ("other", "mean", "xspk")}
                if not all(p.exists() for p in paths.values()):
                    continue
                S = {m: mrt.load_baseline(p) for m, p in paths.items()}
                keys = set(S["other"]["per"])
                if not all(set(s["per"]) == keys for s in S.values()):
                    raise SystemExit(f"{sk}/{mk}/{view}: the three rows cover different utterances")
                lines += [f"## {sname}: {mk}, {view} (n={len(keys)})", "",
                          "| enrolment | " + " | ".join(n for _, n in KEYS) + " | "
                          + " | ".join(f"Δ{n} vs other" for _, n in KEYS) + " |",
                          "|---" * (1 + 2 * len(KEYS)) + "|"]
                for m in ("other", "mean", "xspk"):
                    d = [mrt.paired(S[m], S["other"], k) if m != "other" else "" for k, _ in KEYS]
                    lines.append(f"| {m} | " + " | ".join(_abs(S[m], k) for k, _ in KEYS) + " | "
                                 + " | ".join(d) + " |")
                bp = ROOT / sdir / f"{base}.json"
                if bp.exists():
                    B = mrt.load_baseline(bp)
                    B["per"] = {u: v for u, v in B["per"].items() if u in keys}
                    lines += ["", f"No enrolment vs {bname} (neither has side information), "
                              f"same utterances: " + ", ".join(
                                  f"Δ{n} {mrt.paired(S['mean'], B, k)}" for k, n in KEYS)]
                lines.append("")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
