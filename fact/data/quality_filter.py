"""Clean-speech subsets of an Emilia manifest by the DNSMOS score Emilia ships
in each utterance's .json sidecar (field `dnsmos`, P.835 OVRL).

For training the waveform generator on clean targets: the tokenizer keeps
seeing all of Emilia, only the decoder's training data is selected.

    python -m fact.data.quality_filter scan --manifest <data>/manifests/train.tsv \
        --out <data>/manifests/quality --part 0 --num-parts 16
    python -m fact.data.quality_filter merge --dir <data>/manifests/quality --num-parts 16

scan: for manifest lines [part::num_parts] reads <audio>.json, writes
  part_<k>.tsv (manifest line + TAB dnsmos + TAB duration). A sidecar
  without the field is an error (never treated as clean or dirty).
merge: per threshold in --thresholds writes train_dnsmos<thr>.tsv (the
  original manifest lines) and report.json (hours kept per threshold and a
  histogram), after checking every part is present and complete.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

THRESHOLDS = (3.0, 3.2, 3.3, 3.4, 3.5, 3.6)


def read_quality(audio_path: str, field: str = "dnsmos") -> tuple[float, float]:
    meta = Path(audio_path).with_suffix(".json")
    d = json.loads(meta.read_text(encoding="utf-8"))
    if field not in d:
        raise KeyError(f"{meta}: no {field!r} field (keys {sorted(d)[:8]})")
    dur = d.get("duration")
    return float(d[field]), (float(dur) if dur is not None else float("nan"))


def scan(manifest: Path, out: Path, part: int, num_parts: int, threads: int = 32,
         field: str = "dnsmos") -> dict:
    with open(manifest, encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for i, ln in enumerate(fh) if i % num_parts == part and ln.strip()]
    paths = [ln.split("\t", 1)[0] for ln in lines]
    errors: dict[str, str] = {}

    def one(p):
        try:
            return read_quality(p, field), None
        except Exception as e:  # recorded, the merge refuses incomplete parts
            return None, f"{type(e).__name__}: {str(e)[:200]}"
    with ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(one, paths))
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"part_{part:03d}.tsv.{os.getpid()}.partial"
    n_ok = 0
    with open(tmp, "w", encoding="utf-8") as fh:
        for ln, p, (q, err) in zip(lines, paths, res):
            if err:
                errors[p] = err
                continue
            fh.write(f"{ln}\t{q[0]:.4f}\t{q[1]:.3f}\n")
            n_ok += 1
    os.replace(tmp, out / f"part_{part:03d}.tsv")
    rep = {"part": part, "num_parts": num_parts, "lines": len(lines), "ok": n_ok,
           "errors": len(errors), "error_examples": dict(list(errors.items())[:20])}
    (out / f"part_{part:03d}.json").write_text(json.dumps(rep, indent=1))
    return rep


def merge(d: Path, num_parts: int, thresholds=THRESHOLDS, max_error_frac: float = 0.001) -> dict:
    reps = []
    for k in range(num_parts):
        rp = d / f"part_{k:03d}.json"
        if not rp.exists() or not (d / f"part_{k:03d}.tsv").exists():
            raise SystemExit(f"part {k} missing in {d}")
        reps.append(json.loads(rp.read_text()))
    lines = sum(r["lines"] for r in reps)
    errs = sum(r["errors"] for r in reps)
    if errs > max_error_frac * lines:
        raise SystemExit(f"{errs} of {lines} sidecars unreadable (> {max_error_frac:.1%}); "
                         f"e.g. {next(iter(reps[0]['error_examples'].items()), None)}")
    outs = {t: open(d / f".train_dnsmos{t:.1f}.tsv.{os.getpid()}.partial", "w", encoding="utf-8")
            for t in thresholds}
    hours = {t: 0.0 for t in thresholds}
    kept = {t: 0 for t in thresholds}
    hist = [0] * 50  # 0.1-wide bins over [0, 5)
    total_h = 0.0
    for k in range(num_parts):
        with open(d / f"part_{k:03d}.tsv", encoding="utf-8") as fh:
            for ln in fh:
                *orig, q, dur = ln.rstrip("\n").split("\t")
                q, dur = float(q), float(dur)
                h = dur / 3600 if dur == dur else 0.0
                total_h += h
                hist[min(49, max(0, int(q * 10)))] += 1
                for t in thresholds:
                    if q >= t:
                        outs[t].write("\t".join(orig) + "\n")
                        hours[t] += h
                        kept[t] += 1
    for t, fh in outs.items():
        fh.close()
        os.replace(fh.name, d / f"train_dnsmos{t:.1f}.tsv")
    rep = {"lines": lines, "unreadable": errs, "total_hours": round(total_h, 1),
           "kept": {f"{t:.1f}": {"utterances": kept[t], "hours": round(hours[t], 1)} for t in thresholds},
           "histogram_0.1": hist}
    (d / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m fact.data.quality_filter")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--part", type=int, default=None, help="default $SLURM_ARRAY_TASK_ID")
    s.add_argument("--num-parts", type=int, default=1)
    s.add_argument("--threads", type=int, default=32)
    m = sub.add_parser("merge")
    m.add_argument("--dir", type=Path, required=True)
    m.add_argument("--num-parts", type=int, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "scan":
        part = a.part if a.part is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        if not 0 <= part < a.num_parts or not a.manifest.is_file():
            ap.error("bad --part or missing --manifest")
        print(json.dumps(scan(a.manifest, a.out, part, a.num_parts, a.threads), indent=1))
    else:
        rep = merge(a.dir, a.num_parts)
        print(json.dumps({k: v for k, v in rep.items() if k != "histogram_0.1"}, indent=1))


if __name__ == "__main__":
    main()
