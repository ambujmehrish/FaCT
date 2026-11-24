"""Real-time factor of full encode+decode round trips, same utterances for
every system, on each requested device (streaming claim, docs/ICML_TODO.md).

    python scripts/measure_rtf.py --eval-dir <eval>/librispeech_test_clean_full --n 100 \
        --devices cuda,cpu --cpu-threads 8 --out results/rtf/rtf.json \
        --systems "fact_wave:decoder=<runs>/wavedec_main_c8_rd05_la4/step_0250000.pt:view=full:label=fact_la4" \
                  "focalcodec:config=lucadellalib/focalcodec_50hz_2k_causal:label=focalcodec_stream" \
                  "zipcodec:label=zipcodec" "mimi:n_streams=2:label=mimi2"

RTF = model compute time / audio duration, offline (whole utterance per
call, batch 1, after two warm-up calls), median and mean over utterances.
It measures compute cost, not the algorithmic latency, which is a property
of the architecture (fact/eval/latency.py; the baselines' own papers).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--eval-dir", type=Path)
    src.add_argument("--eval-manifest", type=Path)
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--devices", default="cuda,cpu")
    ap.add_argument("--cpu-threads", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.devices = [d for d in a.devices.split(",") if d]
    if not a.devices or set(a.devices) - {"cuda", "cpu"}:
        ap.error("--devices: cuda and/or cpu")
    if a.n < 1 or a.cpu_threads < 1:
        ap.error("--n and --cpu-threads must be >= 1")
    if not (a.eval_dir or a.eval_manifest).exists():
        ap.error("eval set not found")
    return a


def time_system(tok, wavs, device: str) -> dict:
    from fact.eval.metrics_audio import resample

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
    xs = [(resample(w.float(), sr, tok.sample_rate) if sr != tok.sample_rate else w.float())[None]
          for _, w, sr in wavs]
    for x in xs[:2]:  # warm-up
        tok.reconstruct(x.to(device))
    rtf, dur = [], 0.0
    for x in xs:
        x = x.to(device)
        sync()
        t = time.perf_counter()
        with torch.no_grad():
            tok.reconstruct(x)
        sync()
        sec = x.shape[-1] / tok.sample_rate
        rtf.append((time.perf_counter() - t) / sec)
        dur += sec
    return {"rtf_median": statistics.median(rtf), "rtf_mean": statistics.fmean(rtf),
            "n": len(rtf), "audio_seconds": dur}


def main(argv=None) -> None:
    a = parse_args(argv)
    from fact.baselines import base
    from fact.eval.eval_set import load_eval_manifest, load_eval_wavs
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_baselines import parse_spec, spec_label  # noqa: E402
    base._autoload()
    wavs = (load_eval_wavs(a.eval_dir, a.n) if a.eval_dir
            else load_eval_manifest(a.eval_manifest, a.n, 24_000))
    res = {"n_utterances": len(wavs), "cpu_threads": a.cpu_threads, "results": {}}
    for device in a.devices:
        if device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("--devices cuda but no GPU")
        if device == "cpu":
            torch.set_num_threads(a.cpu_threads)
        for spec in a.systems:
            name, kw = parse_spec(spec)
            kw = {k: v for k, v in kw.items() if k != "label"}
            tok = base.build(name, device=device, **kw)
            r = time_system(tok, wavs, device)
            r["latency_ms_reported"] = getattr(tok, "latency_ms", None)
            res["results"][f"{spec_label(spec)}@{device}"] = r
            print(f"{spec_label(spec)}@{device}: RTF median {r['rtf_median']:.3f}", flush=True)
            del tok
            if device == "cuda":
                torch.cuda.empty_cache()
    from fact.provenance import provenance
    res["provenance"] = provenance()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(f"{a.out.name}.{os.getpid()}.partial")
    tmp.write_text(json.dumps(res, indent=1))
    os.replace(tmp, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
