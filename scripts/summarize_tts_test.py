"""Summarise the zero-shot TTS test (scripts/eval_tts.py shards) per LM.

    python scripts/summarize_tts_test.py --inputs 'results/tts/test_ls_{codec}_shard*.json' \
        --codecs fact mimi2 [--out results/tts/test_ls_summary.json]

On the utterances every LM generated:
- corpus WER vs the target transcript, mean UTMOS, mean SIM to the prompt,
  failure rate, median duration ratio;
- the first codec minus the second, for each.
95% CIs by paired bootstrap with the speaker as the resampling unit: every
target of a speaker shares one prompt, so utterances are not independent.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_tts_prosody import ci, load  # noqa: E402


def stats(per: dict, keys: list, idx) -> dict:
    rows = [per[keys[i]] for i in idx]
    w = sum(r["tts/wer_words"] for r in rows)
    out = {"wer": 100 * sum(r["tts/wer_errors"] for r in rows) / w if w else float("nan")}
    for k, name in (("tts/utmos", "utmos"), ("tts/sim_prompt", "sim"), ("tts/failed", "failed")):
        v = [r[k] for r in rows if r.get(k) is not None]
        out[name] = sum(v) / len(v) if v else float("nan")
    return out


def summarize(pers: dict, n_boot: int = 1000, seed: int = 0) -> dict:
    from fact.eval.eval_set import speaker_key
    codecs = list(pers)
    keys = sorted(set.intersection(*(set(p) for p in pers.values())))
    if not keys:
        raise SystemExit("no utterance common to all codecs")
    groups: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        groups.setdefault(speaker_key(k), []).append(i)
    clusters = list(groups.values())
    point = {c: stats(pers[c], keys, range(len(keys))) for c in codecs}
    for c in codecs:
        dr = sorted(pers[c][k]["tts/dur_ratio"] for k in keys if pers[c][k].get("tts/dur_ratio") is not None)
        point[c]["median_dur_ratio"] = dr[len(dr) // 2] if dr else float("nan")
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        idx = [i for _ in clusters for i in clusters[rng.randrange(len(clusters))]]
        boots.append({c: stats(pers[c], keys, idx) for c in codecs})
    out = {"utterances": {c: len(pers[c]) for c in codecs}, "shared": len(keys),
           "bootstrap_unit": "speaker", "clusters": len(clusters), "metrics": {}}
    for m in ("wer", "utmos", "sim", "failed"):
        row = {c: {"value": point[c][m], "ci95": ci([b[c][m] for b in boots])} for c in codecs}
        if len(codecs) == 2:
            a, b = codecs
            row[f"{a}-{b}"] = {"value": point[a][m] - point[b][m],
                               "ci95": ci([x[a][m] - x[b][m] for x in boots])}
        out["metrics"][m] = row
    out["median_dur_ratio"] = {c: point[c]["median_dur_ratio"] for c in codecs}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--codecs", nargs="+", default=["fact", "mimi2"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args(argv)
    res = summarize({c: load(a.inputs.format(codec=c)) for c in a.codecs}, a.bootstrap)
    print(f"shared utterances: {res['shared']}; bootstrap over {res['clusters']} speakers")
    for m, row in res["metrics"].items():
        print(f"{m:8s} " + "  ".join(f"{c} {v['value']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"
                                     for c, v in row.items()))
    if a.out:
        tmp = f"{a.out}.{os.getpid()}.partial"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
