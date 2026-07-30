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
from fact.eval.reconstruction import energy_correlation, f0_metrics, mel_distance
from fact.modules.mel import LogMelSpectrogram


def load_eval_wavs(eval_dir: Path, limit: int | None) -> list[tuple[str, torch.Tensor, int]]:
    import soundfile as sf
    paths = sorted(p for p in eval_dir.rglob("*") if p.suffix.lower() in (".wav", ".flac"))
    if limit:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No .wav/.flac under {eval_dir}")
    out = []
    for p in paths:
        wav, sr = sf.read(p, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        out.append((str(p.relative_to(eval_dir)), torch.from_numpy(wav), sr))
    return out


def eval_baseline(name: str, wavs, device: str, mel_cfg: FaCTConfig,
                  wer_hook=None) -> dict:
    tok = baseline_registry.build(name, device=device)
    mel_fn = LogMelSpectrogram(mel_cfg.audio).to(device)
    per_utt = []
    t0 = time.time()
    for rel, wav, sr in wavs:
        ref = resample(wav, sr, tok.sample_rate).to(device)
        hyp = tok.reconstruct(ref.unsqueeze(0)).squeeze(0).float()
        m = UtteranceMetrics()
        try:
            m.pesq_wb = pesq_wb(ref, hyp, tok.sample_rate)
        except Exception:
            pass
        try:
            m.stoi = stoi(ref, hyp, tok.sample_rate)
        except Exception:
            pass
        m.si_sdr = si_sdr(ref.cpu(), hyp.cpu())

        # Mel + prosody correlates at the eval config's mel resolution.
        sr_mel = mel_cfg.audio.sample_rate
        ref_m = resample(ref.cpu(), tok.sample_rate, sr_mel).to(device)
        hyp_m = resample(hyp.cpu(), tok.sample_rate, sr_mel).to(device)
        with torch.no_grad():
            mel_ref = mel_fn(ref_m.unsqueeze(0)).cpu()
            mel_hyp = mel_fn(hyp_m.unsqueeze(0)).cpu()
        m.mel_l1, _ = mel_distance(mel_ref, mel_hyp)
        m.energy_corr = energy_correlation(mel_ref, mel_hyp)
        f0_ref = extract_f0(ref_m.cpu().numpy(), sr_mel, mel_cfg.audio.hop_length,
                            mel_cfg.prosody.f0_min, mel_cfg.prosody.f0_max)
        f0_hyp = extract_f0(hyp_m.cpu().numpy(), sr_mel, mel_cfg.audio.hop_length,
                            mel_cfg.prosody.f0_min, mel_cfg.prosody.f0_max)
        m.f0_rmse_hz, m.voicing_f1 = f0_metrics(
            torch.from_numpy(f0_ref), torch.from_numpy(f0_hyp)
        )
        if wer_hook is not None:
            m.extras.update(wer_hook(rel, ref.cpu(), hyp.cpu(), tok.sample_rate))
        per_utt.append((rel, m))

    agg = aggregate([m for _, m in per_utt])
    return {
        "baseline": name,
        "metadata": {
            "sample_rate": tok.sample_rate,
            "frame_rate_hz": tok.frame_rate_hz,
            "discrete_bps": tok.discrete_bps,
            "vocab_sizes": tok.vocab_sizes,
        },
        "n_utterances": len(per_utt),
        "wall_seconds": round(time.time() - t0, 1),
        "aggregate": agg,
        "per_utterance": {rel: {k: v for k, v in vars(m).items() if v not in (None, {})}
                          for rel, m in per_utt},
    }


def build_wer_hook(model_size: str, device: str):
    """Whisper WER hook (optional; needs pre-downloaded whisper checkpoint)."""
    import whisper  # openai-whisper
    model = whisper.load_model(model_size, device=device)
    from fact.eval.wer import normalized_wer

    def hook(rel, ref, hyp, sr):
        r16 = resample(ref, sr, 16_000).numpy()
        h16 = resample(hyp, sr, 16_000).numpy()
        text_ref = model.transcribe(r16, fp16=(device != "cpu"))["text"]
        text_hyp = model.transcribe(h16, fp16=(device != "cpu"))["text"]
        return {"wer_vs_ref_transcript": normalized_wer(text_ref, text_hyp)}
    return hook


def write_summary(results: list[dict], out_dir: Path) -> str:
    keys = ["pesq_wb", "stoi", "si_sdr", "mel_l1", "f0_rmse_hz", "voicing_f1",
            "energy_corr"]
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
    (out_dir / "summary.md").write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", required=True,
                    help=f"comma-separated; available: {baseline_registry.available()}")
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8,
                    help="CPU thread cap (avoid oversubscription on shared nodes)")
    ap.add_argument("--wer-model", default=None,
                    help="whisper model size to add WER (e.g. large-v3)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but no GPU visible")

    args.out.mkdir(parents=True, exist_ok=True)
    wavs = load_eval_wavs(args.eval_dir, args.limit)
    print(f"{len(wavs)} eval utterances from {args.eval_dir}")
    cfg = FaCTConfig()
    wer_hook = build_wer_hook(args.wer_model, args.device) if args.wer_model else None

    results = []
    for name in args.baselines.split(","):
        name = name.strip()
        print(f"=== {name} ===")
        r = eval_baseline(name, wavs, args.device, cfg, wer_hook)
        (args.out / f"{name}.json").write_text(json.dumps(r, indent=2))
        results.append(r)
        print(json.dumps(r["aggregate"], indent=2))
    print(write_summary(results, args.out))


if __name__ == "__main__":
    main()
