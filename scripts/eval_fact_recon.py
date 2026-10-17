"""FaCT reconstruction in the mel domain, on the same eval set as the baselines.

    python scripts/eval_fact_recon.py --ckpt <step.pt> \
        --eval-dir <wavs> --out results/fact_recon/<run>.json \
        [--nfe 32 --sampler fm] [--device cuda]

FaCT's decoder emits log-mel; until the stage-C vocoder exists there is no
FaCT waveform, so PESQ/STOI/F0 cannot be computed yet. What CAN be compared
today is mel-L1 and energy correlation in the SAME mel space
eval_baselines.py uses for Mimi/DAC (FaCTConfig().audio): the baselines'
waveforms are re-analysed into that mel, FaCT's mel is scored directly.
That makes FaCT's number pre-vocoder - an optimistic bound; the vocoder's
own error in this mel space is measured by scripts/vocoder_ceiling.py
(`mel_l1_fact_space`) and must be quoted next to it.

Views:
  full           all channels (content + prosody tokens + residual)
  discrete_only  content + prosody tokens; residual omitted - the
                 pure-token interface a discrete LM would drive

Sampler: a stage-A checkpoint has only learned plain flow matching (d=0),
so the default is --sampler fm with 32 Euler steps; use --sampler shortcut
--nfe 2 only for stage-B (shortcut-distilled) checkpoints.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fact.config import AudioConfig, FaCTConfig
from fact.eval.eval_set import load_eval_wavs
from fact.eval.metrics_audio import resample
from fact.eval.reconstruction import energy_correlation, mel_distance
from fact.model import FaCT
from fact.modules.mel import LogMelSpectrogram

MEL_FIELDS = ("sample_rate", "n_mels", "n_fft", "hop_length", "win_length", "fmin", "fmax")


def check_mel_space(audio: AudioConfig) -> None:
    """Baselines are scored in FaCTConfig().audio's mel; refuse a mismatch."""
    ref = FaCTConfig().audio
    bad = [f for f in MEL_FIELDS if getattr(audio, f) != getattr(ref, f)]
    if bad:
        raise SystemExit(f"checkpoint mel config differs from the baseline eval's in {bad}; "
                         "mel-L1 would not be comparable")


def bitrate(model: FaCT) -> dict:
    """Nominal bits/s per stream at the token rate."""
    cfg = model.cfg
    tok_rate = cfg.audio.sample_rate / (cfg.audio.hop_length * cfg.encoder.frame_stack)
    bn = model.bottleneck
    out = {"token_rate_hz": tok_rate}
    if bn.content_codebook_size:
        out["content_bps"] = math.log2(bn.content_codebook_size) * tok_rate
    if bn.prosody_codebook_size:
        out["prosody_bps"] = math.log2(bn.prosody_codebook_size) * tok_rate / bn.prosody_rate_divisor
    if bn.residual_down is not None:
        # Capped channel: Shannon capacity bound. KL channel: no bound.
        out["residual_bps"] = (bn.residual_capacity / math.log(2) * tok_rate
                               if bn.residual_capacity > 0 else None)
    out["discrete_bps"] = out.get("content_bps", 0.0) + out.get("prosody_bps", 0.0)
    return out


def mean_std(xs: list[float]) -> dict:
    t = torch.tensor(xs, dtype=torch.float64)
    return {"mean": float(t.mean()), "std": float(t.std()) if len(xs) > 1 else 0.0, "n": len(xs)}


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--sampler", choices=["fm", "shortcut"], default="fm")
    ap.add_argument("--views", default="full,discrete_only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = FaCTConfig.from_dict(ckpt["config"])
    check_mel_space(cfg.audio)
    model = FaCT(cfg)
    model.load_state_dict(ckpt["model"])
    model = model.to(args.device).eval()
    mel_fn = LogMelSpectrogram(cfg.audio).to(args.device)
    s = cfg.encoder.frame_stack
    shortcut = args.sampler == "shortcut"
    views = [v.strip() for v in args.views.split(",")]
    for v in views:
        if v not in ("full", "discrete_only"):
            raise SystemExit(f"unknown view {v!r}")
    if "discrete_only" in views and model.bottleneck.content_codebook_size is None:
        raise SystemExit("discrete_only needs a discrete content stream (not the vae variant)")

    wavs = load_eval_wavs(args.eval_dir, args.limit)
    print(f"{len(wavs)} eval utterances; sampler={args.sampler} nfe={args.nfe}")
    scores = {v: {"mel_l1": [], "mel_rmse": [], "energy_corr": []} for v in views}
    t0 = time.time()
    for i, (rel, wav, sr) in enumerate(wavs):
        x = resample(wav, sr, cfg.audio.sample_rate).to(args.device)
        mel = mel_fn(x.unsqueeze(0))
        mel = mel[:, : (mel.shape[1] // s) * s]
        if mel.shape[1] == 0:
            raise SystemExit(f"{rel}: shorter than one token")
        for v in views:
            torch.manual_seed(args.seed + i)  # same noise for every view
            if v == "full":
                hyp = model.reconstruct(mel, nfe=args.nfe, shortcut=shortcut)
            else:
                out = model.encode_mel(mel)
                hyp = model.detokenize(out.content_indices, out.prosody_indices, ref_mel=mel,
                                       residual=None, nfe=args.nfe, shortcut=shortcut)
            l1, rmse = mel_distance(mel.cpu(), hyp.cpu())
            scores[v]["mel_l1"].append(l1)
            scores[v]["mel_rmse"].append(rmse)
            scores[v]["energy_corr"].append(energy_correlation(mel.cpu(), hyp.cpu()))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(wavs)} ({time.time() - t0:.0f}s)")

    report = {
        "ckpt": args.ckpt, "step": ckpt.get("step"),
        "sampler": args.sampler, "nfe": args.nfe, "n_utterances": len(wavs),
        "bitrate": bitrate(model),
        "note": "pre-vocoder mel; quote with vocoder_ceiling mel_l1_fact_space",
        "views": {v: {k: mean_std(xs) for k, xs in m.items()} for v, m in scores.items()},
        "config": {"bottleneck": dataclasses.asdict(cfg.bottleneck),
                   "leakage": dataclasses.asdict(cfg.leakage)},
    }
    text = json.dumps(report, indent=2)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
