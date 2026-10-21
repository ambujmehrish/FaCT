"""Reconstruction evaluation of public-checkpoint baselines (Tier 2).

GPU-first: model inference runs on --device (default cuda); only the C
metric routines (PESQ/STOI) and pyworld F0 run on CPU, one utterance at a
time (light). Threads are capped so shared-node CPU limits are respected.

Run on a compute node with pre-downloaded checkpoints + eval set
(see scripts/download_checkpoints.py, scripts/prepare_eval_set.py,
docs/CINECA.md, slurm/eval_baselines.sbatch):

    python scripts/eval_baselines.py --baselines mimi,dac \
        --eval-dir $WORK/data/eval/librispeech_test_clean \
        --out results/baseline_recon --device cuda

Outputs <out>/<baseline>.json (per-utterance + aggregate) and
<out>/summary.md (one table row per baseline, with bitrate metadata read
from the checkpoints themselves).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fact.baselines import base as baseline_registry
from fact.config import FaCTConfig
from fact.data.prosody_targets import extract_f0
from fact.eval.metrics_audio import (
    UtteranceMetrics, aggregate, pesq_wb, resample, si_sdr, stoi,
)
from fact.eval.reconstruction import band_limited_bins, energy_correlation, f0_metrics, mel_distance
from fact.eval.eval_set import load_eval_manifest, load_eval_wavs, load_transcripts
from fact.modules.mel import LogMelSpectrogram


def parse_spec(spec: str) -> tuple[str, dict]:
    """"mimi:n_streams=2" -> ("mimi", {"n_streams": 2}); bare names pass through."""
    name, _, rest = spec.strip().partition(":")
    kwargs = {}
    for item in filter(None, rest.split(":")):
        k, _, v = item.partition("=")
        kwargs[k] = int(v) if v.lstrip("-").isdigit() else v
    return name, kwargs


def spec_label(spec: str) -> str:
    """Filesystem-safe label: "mimi:n_streams=2" -> "mimi_n_streams2"."""
    return spec.strip().replace(":", "_").replace("=", "")


def eval_baseline(spec: str, wavs, device: str, mel_cfg: FaCTConfig,
                  wer_hook=None, f0_method: str = "pyworld") -> dict:
    name, kwargs = parse_spec(spec)
    tok = baseline_registry.build(name, device=device, **kwargs)
    mel_fn = LogMelSpectrogram(mel_cfg.audio).to(device)
    per_utt = []
    # PESQ/STOI can fail on individual pathological utterances (e.g. pure
    # silence) - tolerated but COUNTED; systematic failure aborts (below).
    failures: dict[str, int] = {}
    successes: dict[str, int] = {}
    last_error: dict[str, str] = {}
    EARLY_ABORT_AFTER = 10  # a metric failing on the FIRST N utterances is systematic

    def attempt(metric: str, fn):
        try:
            v = fn()
            successes[metric] = successes.get(metric, 0) + 1
            return v
        except Exception as e:
            failures[metric] = failures.get(metric, 0) + 1
            last_error[metric] = f"{type(e).__name__}: {e}"
            if not successes.get(metric) and failures[metric] >= EARLY_ABORT_AFTER:
                raise RuntimeError(
                    f"[{name}] {metric} failed on the first "
                    f"{failures[metric]} utterances - systematic setup bug, "
                    f"aborting instead of burning the run. Last error: "
                    f"{last_error[metric]}"
                )
            return None

    t0 = time.time()
    for rel, wav, sr in wavs:
        ref = resample(wav, sr, tok.sample_rate).to(device)
        hyp = tok.reconstruct(ref.unsqueeze(0)).squeeze(0).float()
        m = UtteranceMetrics()
        m.pesq_wb = attempt("pesq_wb", lambda: pesq_wb(ref, hyp, tok.sample_rate))
        m.stoi = attempt("stoi", lambda: stoi(ref, hyp, tok.sample_rate))
        m.si_sdr = si_sdr(ref.cpu(), hyp.cpu())

        # Mel + prosody correlates at the eval config's mel resolution.
        sr_mel = mel_cfg.audio.sample_rate
        ref_m = resample(ref.cpu(), tok.sample_rate, sr_mel).to(device)
        hyp_m = resample(hyp.cpu(), tok.sample_rate, sr_mel).to(device)
        with torch.no_grad():
            mel_ref = mel_fn(ref_m.unsqueeze(0)).cpu()
            mel_hyp = mel_fn(hyp_m.unsqueeze(0)).cpu()
        m.mel_l1, _ = mel_distance(mel_ref, mel_hyp)
        nb = band_limited_bins(mel_cfg.audio, sr)
        m.extras["mel_l1_band"] = mel_distance(mel_ref[..., :nb], mel_hyp[..., :nb])[0]
        m.energy_corr = energy_correlation(mel_ref, mel_hyp)
        f0_ref = extract_f0(ref_m.cpu().numpy(), sr_mel, mel_cfg.audio.hop_length,
                            mel_cfg.prosody.f0_min, mel_cfg.prosody.f0_max,
                            method=f0_method)
        f0_hyp = extract_f0(hyp_m.cpu().numpy(), sr_mel, mel_cfg.audio.hop_length,
                            mel_cfg.prosody.f0_min, mel_cfg.prosody.f0_max,
                            method=f0_method)
        m.f0_rmse_hz, m.voicing_f1 = f0_metrics(
            torch.from_numpy(f0_ref), torch.from_numpy(f0_hyp)
        )
        if wer_hook is not None:
            m.extras.update(wer_hook(rel, hyp.cpu(), tok.sample_rate))
        per_utt.append((rel, m))

    for metric, count in failures.items():
        if count == len(wavs):
            raise RuntimeError(
                f"[{name}] {metric} failed on ALL {count} utterances - this "
                f"is a harness/setup bug, not a data quirk. Last error: "
                f"{last_error[metric]}"
            )
        print(f"WARNING [{name}]: {metric} failed on {count}/{len(wavs)} "
              f"utterances (last: {last_error[metric]})")

    agg = aggregate([m for _, m in per_utt])
    return {
        "baseline": spec_label(spec),
        "metadata": {
            "sample_rate": tok.sample_rate,
            "frame_rate_hz": tok.frame_rate_hz,
            "discrete_bps": tok.discrete_bps,
            "vocab_sizes": tok.vocab_sizes,
        },
        "n_utterances": len(per_utt),
        "metric_failures": failures,
        "wall_seconds": round(time.time() - t0, 1),
        "aggregate": agg,
        "per_utterance": {rel: {k: v for k, v in vars(m).items() if v not in (None, {})}
                          for rel, m in per_utt},
    }


def build_wer_hook(model_size: str, device: str, transcripts: dict[str, str]):
    """Whisper WER of the reconstruction against the GROUND-TRUTH transcript
    (codec-paper standard), with Whisper's English normalizer. Errors and
    reference word counts are stored so tables report corpus-level WER."""
    import os

    import whisper  # openai-whisper
    from whisper.normalizers import EnglishTextNormalizer
    from fact.eval.wer import word_edit_distance
    if not transcripts:
        raise SystemExit("--wer-model needs ground-truth transcripts (.txt next to wavs, "
                         "or a manifest with a text column)")
    cache = os.environ.get("FACT_CACHE")
    root = str(Path(cache) / "whisper") if cache else None  # offline: $FACT_CACHE
    model = whisper.load_model(model_size, device=device, download_root=root)
    norm = EnglishTextNormalizer()

    def hook(rel, hyp, sr):
        if rel not in transcripts:
            raise KeyError(f"no ground-truth transcript for {rel}")
        h16 = resample(hyp, sr, 16_000).numpy()
        text = model.transcribe(h16, language="en", fp16=(device != "cpu"))["text"]
        ref_w, hyp_w = norm(transcripts[rel]).split(), norm(text).split()
        err = word_edit_distance(ref_w, hyp_w)
        return {"wer_errors": float(err), "wer_words": float(len(ref_w)),
                "wer": err / max(1, len(ref_w))}
    return hook


def write_summary(results: list[dict], out_dir: Path) -> str:
    keys = ["pesq_wb", "stoi", "si_sdr", "mel_l1", "mel_l1_band", "f0_rmse_hz",
            "voicing_f1", "energy_corr", "wer"]
    lines = [
        "| baseline | fps | kbps | " + " | ".join(keys) + " | n |",
        "|" + "---|" * (len(keys) + 4),
    ]
    for r in results:
        md = r["metadata"]
        bps = md["discrete_bps"]
        cells = [r["baseline"], f"{md['frame_rate_hz']:.1f}",
                 f"{bps / 1000:.2f}" if bps else "-"]
        for k in keys:
            a = r["aggregate"].get(k)
            cells.append(f"{a['mean']:.3f}±{a['std']:.3f}" if a else "-")
        cells.append(str(r["n_utterances"]))
        lines.append("| " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", required=True,
                    help="comma-separated specs, name[:key=value...], e.g. "
                         "mimi:n_streams=2,dac:n_quantizers=2; available: "
                         f"{baseline_registry.available()}")
    ap.add_argument("--eval-dir", type=Path, default=None,
                    help="directory of wav/flac (+ .txt transcripts)")
    ap.add_argument("--eval-manifest", type=Path, default=None,
                    help="path<TAB>text manifest instead (e.g. Emilia test.tsv, mp3)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8,
                    help="CPU thread cap (avoid oversubscription on shared nodes)")
    ap.add_argument("--wer-model", default=None,
                    help="whisper model size to add WER (e.g. large-v3)")
    ap.add_argument("--f0", default="pyworld", choices=["pyworld", "autocorr"],
                    help="F0 extractor for prosody metrics (no silent fallback)")
    ap.add_argument("--summary-only", action="store_true",
                    help="rebuild summary.md from the per-baseline JSONs in --out")
    args = ap.parse_args()

    if args.summary_only:
        results = []
        for name in args.baselines.split(","):
            path = args.out / f"{spec_label(name)}.json"
            if not path.exists():
                raise SystemExit(f"missing {path}: run the eval for {name.strip()} first")
            results.append(json.loads(path.read_text(encoding="utf-8")))
        print(write_summary(results, args.out))
        return

    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but no GPU visible")
    if args.f0 == "pyworld":
        try:
            import pyworld  # noqa: F401
        except ImportError:
            raise SystemExit(
                "pyworld is not installed (pip install -e '.[audio]') - "
                "failing before model loading rather than mid-eval. Pass "
                "--f0 autocorr only to knowingly use the coarse extractor."
            )

    args.out.mkdir(parents=True, exist_ok=True)
    cfg = FaCTConfig()
    if (args.eval_dir is None) == (args.eval_manifest is None):
        raise SystemExit("give exactly one of --eval-dir / --eval-manifest")
    if args.eval_manifest:
        wavs = load_eval_manifest(args.eval_manifest, args.limit, cfg.audio.sample_rate)
    else:
        wavs = load_eval_wavs(args.eval_dir, args.limit)
    transcripts = load_transcripts(args.eval_dir, args.eval_manifest)
    print(f"{len(wavs)} eval utterances from {args.eval_manifest or args.eval_dir} "
          f"({len(transcripts)} transcripts)")
    wer_hook = build_wer_hook(args.wer_model, args.device, transcripts) if args.wer_model else None

    results = []
    for name in args.baselines.split(","):
        name = name.strip()
        print(f"=== {name} ===")
        r = eval_baseline(name, wavs, args.device, cfg, wer_hook, f0_method=args.f0)
        (args.out / f"{spec_label(name)}.json").write_text(json.dumps(r, indent=2),
                                                         encoding="utf-8")
        results.append(r)
        print(json.dumps(r["aggregate"], indent=2))
    print(write_summary(results, args.out))


if __name__ == "__main__":
    main()
