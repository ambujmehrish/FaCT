"""Summarise the TTS prosody-control test (scripts/eval_tts_prosody.py shards).

    python scripts/summarize_tts_prosody.py --inputs 'results/tts/prosody_ls_{codec}_shard*.json' \
        --codecs fact mimi2 [--out results/tts/prosody_ls_summary.json]

Per LM on the utterances every LM generated:
  - corpus WER per group vs the ground truth (base / fixed / free);
  - word changes vs the base sample (fixed_vs_base / free_vs_base);
  - pitch diversity: s.d. of the median F0 (semitones) and of the log-F0
    spread (cents) across base + samples;
  - frame-aligned contour distance between the fixed samples (cents);
  - duration s.d. (s).
And two within-LM ratios that measure control:
  - pitch_kept = fixed / free median-F0 s.d. (1 = holding stream 0 keeps all
    of free sampling's pitch variation);
  - words_moved = fixed_vs_base / free_vs_base word changes (lower = holding
    stream 0 keeps the words).
95% CIs by paired bootstrap over utterances, and for each metric the first
codec minus the second.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random

MEANS = ["fixed/median_st_sd", "free/median_st_sd", "fixed/spread_sd_cents", "free/spread_sd_cents",
         "fixed/contour_cents", "fixed/dur_sd_s", "free/dur_sd_s"]
WERS = {"wer_base": "base", "wer_fixed": "fixed", "wer_free": "free",
        "words_vs_base_fixed": "fixed_vs_base", "words_vs_base_free": "free_vs_base"}


def load(pattern: str) -> dict:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files match {pattern}")
    per, shards = {}, set()
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        shards.add(d["shard"])
        n = d["num_shards"]
        per.update(d["per_utterance"])
    if shards != set(range(n)):
        raise SystemExit(f"{pattern}: shards {sorted(shards)} of {n}")
    return per


def stats(per: dict, keys: list, idx) -> dict:
    rows = [per[keys[i]] for i in idx]

    def mean(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return sum(v) / len(v) if v else float("nan")

    def wer(g):
        w = sum(r.get(f"{g}/wer_words", 0.0) for r in rows)
        return 100 * sum(r.get(f"{g}/wer_errors", 0.0) for r in rows) / w if w else float("nan")

    out = {k: mean(k) for k in MEANS} | {name: wer(g) for name, g in WERS.items()}
    out["pitch_kept"] = out["fixed/median_st_sd"] / out["free/median_st_sd"]
    out["words_moved"] = out["words_vs_base_fixed"] / out["words_vs_base_free"]
    return out


def ci(values: list) -> list:
    v = sorted(x for x in values if x == x)
    return [v[int(0.025 * len(v))], v[min(len(v) - 1, int(0.975 * len(v)))]] if v else [None, None]


def summarize(pers: dict, n_boot: int = 1000, seed: int = 0) -> dict:
    codecs = list(pers)
    keys = sorted(set.intersection(*(set(p) for p in pers.values())))
    if not keys:
        raise SystemExit("no utterance common to all codecs")
    point = {c: stats(pers[c], keys, range(len(keys))) for c in codecs}
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(len(keys)) for _ in keys]
        boots.append({c: stats(pers[c], keys, idx) for c in codecs})
    out = {"utterances": {c: len(pers[c]) for c in codecs}, "shared": len(keys), "metrics": {}}
    for m in point[codecs[0]]:
        row = {c: {"value": point[c][m], "ci95": ci([b[c][m] for b in boots])} for c in codecs}
        if len(codecs) == 2:
            a, b = codecs
            row[f"{a}-{b}"] = {"value": point[a][m] - point[b][m],
                               "ci95": ci([x[a][m] - x[b][m] for x in boots])}
        out["metrics"][m] = row
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="glob with {codec}, e.g. results/tts/prosody_ls_{codec}_shard*.json")
    ap.add_argument("--codecs", nargs="+", default=["fact", "mimi2"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args(argv)
    res = summarize({c: load(a.inputs.format(codec=c)) for c in a.codecs}, a.bootstrap)
    print(f"shared utterances: {res['shared']} ({res['utterances']})")
    for m, row in res["metrics"].items():
        cells = "  ".join(f"{c} {v['value']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]" for c, v in row.items())
        print(f"{m:24s} {cells}")
    if a.out:
        tmp = f"{a.out}.{os.getpid()}.partial"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
