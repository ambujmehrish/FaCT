"""Pick a TTS LM's sampling setting from its Emilia-dev runs (rule fixed in
advance): the lowest corpus WER; ties (within 0.1 WER points) go to the
higher speaker similarity to the prompt. All settings are compared on the
utterances every one of them generated.

    python scripts/select_tts_sampling.py results/tts/dev_fact_*.json
Prints a table to stderr and "TEMPERATURE TOP_K TOP_P" to stdout.
"""

from __future__ import annotations

import json
import sys


def main(paths: list[str]) -> None:
    runs = [json.load(open(p, encoding="utf-8")) for p in paths]
    if not runs:
        sys.exit("no dev result files")
    if len({r["lm"] for r in runs}) != 1:
        sys.exit("files come from different LMs")
    common = set.intersection(*(set(r["per_utterance"]) for r in runs))
    rows = []
    for p, r in zip(paths, runs):
        per = [r["per_utterance"][k] for k in sorted(common)]
        e = sum(m.get("tts/wer_errors", 0.0) for m in per)
        w = sum(m.get("tts/wer_words", 0.0) for m in per)
        sims = [m["tts/sim_prompt"] for m in per if "tts/sim_prompt" in m]
        s = r["sampling"]
        rows.append((100 * e / max(w, 1), sum(sims) / max(len(sims), 1), s, p))
    best_wer = min(x[0] for x in rows)
    best = max((x for x in rows if x[0] <= best_wer + 0.1), key=lambda x: x[1])
    print(f"{len(common)} common dev utterances", file=sys.stderr)
    for wer, sim, s, p in sorted(rows, key=lambda x: x[0]):
        tag = " <- selected" if (wer, sim, s, p) == best else ""
        print(f"  T={s['temperature']} top_k={s['top_k']} top_p={s['top_p']}  WER {wer:.2f}  SIM {sim:.3f}{tag}",
              file=sys.stderr)
    s = best[2]
    print(s["temperature"], s["top_k"], s["top_p"])


if __name__ == "__main__":
    main(sys.argv[1:])
