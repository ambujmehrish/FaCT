"""Reconstruction table: baselines + FaCT views on one eval set, with CIs.

    python scripts/make_recon_table.py \
        --baselines results/baseline_recon_ls/*.json \
        --fact fact_main=results/fact_recon/fact_main_ls_spkother.json \
        --title "LibriSpeech test-clean" --out results/tables/recon_ls.md \
        [--source-sr 16000] [--paired <row name> ...]

Every number is recomputed from per-utterance values:
  - Mean with a bootstrap 95% CI over utterances: 2,000 resamples, seeded per
    cell from (row, column), so a cell's CI does not depend on row order or on
    which other rows are in the table.
  - WER is corpus-level (total errors / total reference words), with the CI
    from resampling utterances.
  - Paired tables give per-utterance differences on the same utterances, so
    utterance difficulty cancels. Paired WER is the difference of corpus WERs
    under the same resample.
Rows named "FaCT <run> (...)" are mel-domain decoder outputs (before the
vocoder): waveform metrics (PESQ/STOI/WER/F0) are "n/a" for them, never filled
from elsewhere. Vocoded and stage-D FaCT results come in as fact_* / factwave_*
rows.

--source-sr: sample rate of the eval audio. A system whose output rate is
lower is scored against a reference band-limited to its own Nyquist
(scripts/eval_baselines.py), so its mel-L1 and energy r are not comparable
with full-band systems. Those cells show "n/c".

Bitrates: a row whose bitrate exceeds its token bitrate (vocab bits x frame
rate) carries a capped continuous residual. It is shown as
"tokens + <=residual", because the residual is a capacity bound, not a coded
bitstream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
BAND_SENSITIVE = {"mel_l1_band", "energy_corr"}  # depend on content above the codec's Nyquist
PAIRED_COLS = [("wer", "ΔWER %"), ("utmos", "ΔUTMOS"), ("spk_sim", "ΔSIM"), ("mel_l1_band", "Δmel-L1")]

CHUNK = 100  # resamples per batch: keeps peak memory small (login nodes)


def cell_rng(*parts: str) -> np.random.Generator:
    """A fixed generator per (row, column[, other row]): stable across tables."""
    h = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


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


def boot_ratio_diff(ea, eb, words, rng) -> tuple[float, float, float]:
    """Corpus WER(a) - WER(b) on the same utterances (same reference words)."""
    r = np.concatenate([(ea[idx].sum(axis=1) - eb[idx].sum(axis=1)) / np.maximum(words[idx].sum(axis=1), 1)
                        for idx in _resample_idx(len(ea), rng)])
    return (float((ea.sum() - eb.sum()) / max(words.sum(), 1)),
            float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)))


def token_bps(meta: dict) -> float | None:
    vs, fr = meta.get("vocab_sizes"), meta.get("frame_rate_hz")
    if not vs or not fr:
        return None
    return float(fr) * sum(math.log2(v) for v in vs)


def load_baseline(path: Path) -> dict:
    r = json.loads(path.read_text(encoding="utf-8"))
    per = {}
    for utt, m in r["per_utterance"].items():
        row = {k: v for k, v in m.items() if k != "extras" and isinstance(v, (int, float))}
        row.update({k: v for k, v in (m.get("extras") or {}).items()})
        per[utt] = row
    meta = r["metadata"]
    bps, tok = meta.get("discrete_bps"), token_bps(meta)
    res = (bps - tok) if (bps is not None and tok is not None and bps - tok > 1.0) else None
    return {"name": r["baseline"], "kbps": bps / 1000 if bps else None,
            "tok_kbps": (tok / 1000 if res is not None else None),
            "res_kbps": (res / 1000 if res is not None else None),
            "sr": meta.get("sample_rate"), "per": per, "fact": False}


def load_fact(label: str, path: Path) -> list[dict]:
    r = json.loads(path.read_text(encoding="utf-8"))
    br = r["bitrate"]
    disc = br.get("discrete_bps", 0.0)
    res = br.get("residual_bps")
    # Tokens-only mode the model was trained for: residual-dropout models saw
    # "no residual" in training (discrete_only); others only the channel prior.
    dropout = (r.get("config", {}).get("bottleneck", {}) or {}).get("residual_dropout", 0.0)
    native = "discrete_only" if dropout and "discrete_only" in r["views"] else "tokens_prior"
    spk = r.get("spk_ref") or "full"
    rows = []
    for view in r["views"]:
        per = {utt: {k.split("/", 1)[1]: v for k, v in m.items() if k.startswith(view + "/")}
               for utt, m in r["per_utterance"].items()}
        full = view == "full"
        note = {"full": "tokens + residual", "tokens_prior": "tokens only",
                "discrete_only": "tokens only (residual removed)"}.get(view, view)
        rows.append({"name": f"FaCT {label} ({note}; speaker ref: {spk})",
                     "kbps": (disc + (res or 0.0)) / 1000 if full else disc / 1000,
                     "tok_kbps": disc / 1000 if (full and res) else None,
                     "res_kbps": res / 1000 if (full and res) else None,
                     "sr": None, "per": per, "fact": True, "unbounded_residual": full and res is None,
                     "label": label, "view": view, "note": note, "spk_ref": spk,
                     "native_tokens": view == native})
    return rows


def kbps_text(s: dict) -> str:
    if s["kbps"] is None:
        return "-"
    if s.get("res_kbps") is not None:
        return f"{s['tok_kbps']:.3f} + ≤{s['res_kbps']:.3f}"
    kb = f"{s['kbps']:.3f}"
    return kb + (" + unbounded residual" if s.get("unbounded_residual") else "")


def not_comparable(s: dict, key: str, source_sr: int | None) -> bool:
    return (key in BAND_SENSITIVE and source_sr is not None and s.get("sr") is not None
            and s["sr"] < source_sr)


def cell(sys: dict, key: str, fmt: str, source_sr: int | None) -> str:
    if not_comparable(sys, key, source_sr):
        return "n/c"
    per = sys["per"]
    rng = cell_rng(sys["name"], key)
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


def paired(a: dict, b: dict, key: str, source_sr: int | None = None) -> str:
    if not_comparable(a, key, source_sr) or not_comparable(b, key, source_sr):
        return "n/c"
    rng = cell_rng(a["name"], b["name"], key)
    if key == "wer":
        shared = [u for u in a["per"] if u in b["per"] and "wer_errors" in a["per"][u]
                  and "wer_errors" in b["per"][u]]
        if len(shared) < 10:
            return "-"
        ea = np.array([a["per"][u]["wer_errors"] for u in shared])
        eb = np.array([b["per"][u]["wer_errors"] for u in shared])
        w = np.array([a["per"][u]["wer_words"] for u in shared])
        m, lo, hi = (100 * v for v in boot_ratio_diff(ea, eb, w, rng))
        fmt = "{:+.2f}"
    else:
        shared = [u for u in a["per"] if u in b["per"] and a["per"][u].get(key) is not None
                  and b["per"][u].get(key) is not None]
        if len(shared) < 10:
            return "-"
        d = np.array([a["per"][u][key] - b["per"][u][key] for u in shared])
        m, lo, hi = boot_mean(d, rng)
        fmt = "{:+.3f}"
    sig = "*" if lo > 0 or hi < 0 else ""
    return f"{fmt.format(m)} [{fmt.format(lo)}, {fmt.format(hi)}]{sig}"


def is_fact_row(s: dict) -> bool:
    return s["fact"] or s["name"].startswith(("fact_", "factwave_"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", nargs="+", type=Path, default=[])
    ap.add_argument("--fact", nargs="*", default=[], metavar="LABEL=JSON")
    ap.add_argument("--title", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ref", default=None,
                    help="FaCT label for the paired mel-L1 table (default: first --fact)")
    ap.add_argument("--source-sr", type=int, default=None,
                    help="eval audio rate; band-sensitive cells of lower-rate systems become n/c")
    ap.add_argument("--paired", nargs="*", default=[],
                    help="row names (e.g. factwave_..._full_spkother) for a paired WER/UTMOS/SIM "
                         "table against every non-FaCT row")
    args = ap.parse_args(argv)

    systems = [load_baseline(p) for p in args.baselines]
    facts = []
    for spec in args.fact:
        label, _, path = spec.partition("=")
        rows = load_fact(label, Path(path))
        if any(r["spk_ref"] != "other" for r in rows):
            raise SystemExit(f"--fact {spec}: speaker reference {rows[0]['spk_ref']!r} is taken from "
                             "the target utterance itself (oracle); use the spk_ref=other result")
        facts += rows
    systems = sorted(systems, key=lambda s: (s["kbps"] or 0, s["name"])) + facts
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

    f0_missing = {s["name"]: sum(1 for m in s["per"].values() if not np.isfinite(m.get("f0_rmse_hz", np.nan)))
                  for s in systems if any("f0_rmse_hz" in m for m in s["per"].values())}
    f0_missing = {k: v for k, v in f0_missing.items() if v}
    lines = [f"## {args.title}", "",
             f"Mean [95% bootstrap CI] over utterances (n = {', '.join(map(str, sorted(n)))}"
             f"{'; every row restricted to the utterances all systems share' if restricted else ''}). "
             "WER is corpus-level vs the ground-truth transcript. mel-L1 uses bins below the "
             "source audio's Nyquist. fact_* rows are vocoded waveforms (stage-C vocoder), "
             "factwave_* rows are stage-D waveform decoders; both go through the same metrics "
             "as the baselines. FaCT rows use the speaker vector of another utterance of the "
             "same speaker (enrolment). Rows named \"FaCT <run> (...)\" are mel-domain decoder "
             "outputs before the vocoder, so they have mel-L1 and energy only. "
             "kbps \"a + ≤b\": a = token bitrate, b = capacity bound of the continuous residual "
             "(not a coded bitstream).", ""]
    if args.source_sr:
        lines += [f"n/c = not comparable: the system outputs below the {args.source_sr} Hz source rate "
                  "and is scored against a reference band-limited to its own Nyquist, so its "
                  "mel-L1 and energy r miss the error above that band.", ""]
    if f0_missing:
        lines += ["F0 RMSE is over utterances with jointly voiced frames; utterances without any: "
                  + ", ".join(f"{k} ({v})" for k, v in sorted(f0_missing.items())) + ".", ""]
    lines += ["| system | kbps | " + " | ".join(h for _, h, _, _ in COLS) + " |",
              "|" + "---|" * (len(COLS) + 2)]
    for s in systems:
        lines.append(f"| {s['name']} | {kbps_text(s)} | " +
                     " | ".join(cell(s, k, f, args.source_sr) for k, _, _, f in COLS) + " |")

    others = [s for s in systems if not is_fact_row(s) and s["name"] != "vocoder_copysyn"]
    by_name = {s["name"]: s for s in systems}
    missing = [p for p in args.paired if p not in by_name]
    if missing:
        raise SystemExit(f"--paired rows not in the table: {missing}")
    for pname in args.paired:
        a = by_name[pname]
        lines += ["", f"**Paired differences, {pname} ({kbps_text(a)} kbps) minus each baseline** "
                  "(per utterance, same audio; for WER the difference of corpus WERs; "
                  "* = 95% CI excludes 0).", "",
                  "| baseline | kbps | " + " | ".join(h for _, h in PAIRED_COLS) + " |",
                  "|" + "---|" * (len(PAIRED_COLS) + 2)]
        for b in others:
            lines.append(f"| {b['name']} | {kbps_text(b)} | " +
                         " | ".join(paired(a, b, k, args.source_sr) for k, _ in PAIRED_COLS) + " |")

    ref_label = args.ref or (facts[0]["label"] if facts else None)
    refs = [f for f in facts if f["label"] == ref_label and (f["view"] == "full" or f["native_tokens"])]
    if args.ref and not refs:
        raise SystemExit(f"--ref {args.ref!r} is not one of the --fact labels")
    if refs and others:
        lines += ["", f"**Paired mel-L1 difference, FaCT {ref_label} (mel domain, before the vocoder) "
                  "minus each baseline** (per utterance, same audio; negative = FaCT better; * = 95% CI "
                  "excludes 0). Tokens-only uses the mode the model was trained for.", "",
                  "| baseline | kbps | " + " | ".join(
                      f"Δ vs {r['note']} ({kbps_text(r)} kbps)" for r in refs) + " |",
                  "|" + "---|" * (len(refs) + 2)]
        for s in others:
            lines.append(f"| {s['name']} | {kbps_text(s)} | " +
                         " | ".join(paired(r, s, "mel_l1_band", args.source_sr) for r in refs) + " |")

    text = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
