"""Tripwire summary table from results/tripwires/*.json (all runs, all steps).

    python scripts/tripwire_table.py --out results/tables/tripwires.md
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

COLS = [("full", "full ↓"), ("R2_swap", "R2_swap"), ("tok_only", "tokens-only Δ ↓"),
        ("swap_pro", "swap prosody Δ"), ("uncond", "uncond Δ"), ("CER", "CER ↓"),
        ("c_voic", "content→voicing"), ("c_pit", "content→pitch"), ("p_pit", "prosody→pitch"),
        ("pit_maj", "pitch majority")]


def row(r: dict) -> dict:
    c, p = r.get("channel_zeroing", {}), r.get("probes", {})
    return {
        "full": c.get("loss", {}).get("none"),
        "R2_swap": c.get("R2_swap"),
        "tok_only": c.get("prior_delta_vs_full", {}).get("residual"),
        "swap_pro": c.get("swap_delta_vs_full", {}).get("prosody"),
        "uncond": c.get("uncond_delta"),
        "CER": r.get("greedy_byte_cer"),
        "c_voic": p.get("content_to_voicing_acc"),
        "c_pit": p.get("content_to_pitch_acc"),
        "p_pit": p.get("prosody_to_pitch_acc"),
        "pit_maj": p.get("pitch_majority_acc"),
        "flags": ",".join(k for k, v in r.get("flags", {}).items() if v) or "-",
    }


def run_and_step(path: Path, r: dict) -> tuple[str, int]:
    """Run name from '<run>_<7 digits>.json', else from the checkpoint's directory
    (tripwires.sbatch's default name is 'step_<7 digits>.json')."""
    m = re.match(r"(.+)_(\d{7})$", path.stem)
    if m and m.group(1) != "step":
        return m.group(1), int(m.group(2))
    ckpt = r.get("ckpt")
    run = Path(ckpt).parent.name if ckpt else path.stem
    return run, int(r.get("step") or (m.group(2) if m else 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/tripwires/*.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    latest: dict[tuple[str, int], tuple[float, dict]] = {}
    for f in glob.glob(args.glob):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        if "R2_swap" not in r.get("channel_zeroing", {}):
            continue  # pre-audit report format; not comparable
        run, step = run_and_step(Path(f), r)
        mtime = Path(f).stat().st_mtime
        if (run, step) not in latest or mtime > latest[(run, step)][0]:
            latest[(run, step)] = (mtime, row(r))  # rerun of the same ckpt: newest wins
    rows = [(run, step, v) for (run, step), (_, v) in sorted(latest.items())]

    lines = ["## Tripwires (speaker-disjoint dev set)", "",
             "Δ = increase in flow-matching loss under the ablation; swap = channel taken from "
             "another utterance; tokens-only = residual message 0 through the channel.", "",
             "| run | step | " + " | ".join(h for _, h in COLS) + " | flags |",
             "|" + "---|" * (len(COLS) + 3)]
    for run, step, v in rows:
        cells = ["-" if v[k] is None else f"{v[k]:.4f}" if k in ("full", "tok_only", "swap_pro", "uncond")
                 else f"{v[k]:.2f}" if k == "R2_swap" else f"{v[k]:.3f}" for k, _ in COLS]
        lines.append(f"| {run} | {step // 1000}K | " + " | ".join(cells) + f" | {v['flags']} |")
    text = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
