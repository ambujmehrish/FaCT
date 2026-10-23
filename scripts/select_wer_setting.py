"""Pick a run's sampling setting from the dev WER sweep (submit_wer_sweep.sh).

Rule (fixed before any result): among the run's full-view settings, the
lowest dev corpus WER whose dev band mel-L1 is at most --mel-slack above
the reference setting's (T=0.5, CFG=1.5 - the mel-L1 sweep winner).
All settings are compared on the utterances they all scored.

    python scripts/select_wer_setting.py --run main_c8_rd05 [--step 0400000]

Prints a table to stderr and "T NFE CFG" of the winner to stdout.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PAT = re.compile(r"_full_T(?P<t>[0-9.]+)_G(?P<g>[0-9.]+)\.json$")


def per_utt(path: Path) -> dict:
    r = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for u, m in r["per_utterance"].items():
        row = {k: v for k, v in m.items() if isinstance(v, (int, float))}
        row.update(m.get("extras") or {})
        out[u] = row
    return out


def summarize(per: dict, utts: list[str]) -> dict:
    err = sum(per[u]["wer_errors"] for u in utts)
    words = sum(per[u]["wer_words"] for u in utts)
    mel = sum(per[u]["mel_l1_band"] for u in utts) / len(utts)
    ut = [per[u]["utmos"] for u in utts if per[u].get("utmos") is not None]
    return {"wer": 100 * err / max(words, 1), "mel": mel,
            "utmos": sum(ut) / len(ut) if ut else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", default="0400000")
    ap.add_argument("--dir", type=Path, default=Path("results/baseline_recon_dev"))
    ap.add_argument("--ref", default="0.5,1.5", help="reference T,CFG")
    ap.add_argument("--mel-slack", type=float, default=0.02)
    ap.add_argument("--nfe", type=int, default=32)
    args = ap.parse_args()
    files = {}
    for f in args.dir.glob(f"fact_{args.run}@{args.step}_full_T*_G*.json"):
        m = PAT.search(f.name)
        if m:
            files[(m["t"], m["g"])] = f
    ref = tuple(args.ref.split(","))
    if ref not in files:
        sys.exit(f"reference setting T={ref[0]} CFG={ref[1]} missing for {args.run}@{args.step}")
    pers = {k: per_utt(f) for k, f in files.items()}
    common = sorted(set.intersection(*(set(p) for p in pers.values())))
    common = [u for u in common if all("wer_errors" in p[u] and "mel_l1_band" in p[u]
                                       for p in pers.values())]
    if not common:
        sys.exit("no utterance has WER and mel-L1 in every setting")
    stats = {k: summarize(p, common) for k, p in pers.items()}
    limit = stats[ref]["mel"] + args.mel_slack
    ok = {k: s for k, s in stats.items() if s["mel"] <= limit}
    best = min(ok, key=lambda k: ok[k]["wer"])
    print(f"{args.run}@{args.step}: {len(files)} settings, {len(common)} common dev utterances; "
          f"mel-L1 limit {limit:.3f}", file=sys.stderr)
    print("  T     CFG   WER %   mel-L1  UTMOS", file=sys.stderr)
    for k in sorted(stats, key=lambda k: (float(k[0]), float(k[1]))):
        s = stats[k]
        tag = " <- selected" if k == best else ("" if k in ok else " (mel-L1 over limit)")
        print(f"  {k[0]:<5} {k[1]:<5} {s['wer']:6.2f}  {s['mel']:.3f}   {s['utmos']:.2f}{tag}",
              file=sys.stderr)
    print(best[0], args.nfe, best[1])


if __name__ == "__main__":
    main()
