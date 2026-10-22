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
  tokens_prior   content + prosody tokens; residual = message 0 through the
                 real channel - the tokens-only interface a discrete LM drives
  discrete_only  content + prosody tokens; residual removed entirely (the
                 in-distribution tokens-only mode of residual-dropout models)

Sampler: a stage-A checkpoint has only learned plain flow matching (d=0),
so the default is --sampler fm with 32 Euler steps; use --sampler shortcut
--nfe 2 only for stage-B (shortcut-distilled) checkpoints.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fact.config import AudioConfig, FaCTConfig
from fact.eval.bitrate import fact_bitrate
from fact.eval.eval_set import load_eval_manifest, load_eval_wavs, reference_partners
from fact.eval.metrics_audio import resample
from fact.eval.reconstruction import band_limited_bins, energy_correlation, mel_distance
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
    return fact_bitrate(model)


def mean_std(xs: list[float]) -> dict:
    t = torch.tensor(xs, dtype=torch.float64)
    return {"mean": float(t.mean()), "std": float(t.std()) if len(xs) > 1 else 0.0, "n": len(xs)}


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval-dir", type=Path, default=None)
    ap.add_argument("--eval-manifest", type=Path, default=None,
                    help="path<TAB>text manifest instead (e.g. Emilia test.tsv)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--sampler", choices=["fm", "shortcut"], default="fm")
    ap.add_argument("--views", default="full,tokens_prior,discrete_only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spk-ref", default="full",
                    help="speaker reference for the decoder: full (the utterance itself, "
                         "not streaming-valid), prefix:<seconds> (its first seconds only), "
                         "or other (a different utterance of the same speaker)")
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
        if v not in ("full", "tokens_prior", "discrete_only"):
            raise SystemExit(f"unknown view {v!r}")
    if set(views) - {"full"} and model.bottleneck.content_codebook_size is None:
        raise SystemExit("token views need a discrete content stream (not the vae variant)")

    if (args.eval_dir is None) == (args.eval_manifest is None):
        raise SystemExit("give exactly one of --eval-dir / --eval-manifest")
    wavs = (load_eval_manifest(args.eval_manifest, args.limit, cfg.audio.sample_rate)
            if args.eval_manifest else load_eval_wavs(args.eval_dir, args.limit))
    print(f"{len(wavs)} eval utterances; sampler={args.sampler} nfe={args.nfe} "
          f"spk_ref={args.spk_ref}")
    mode, _, arg = args.spk_ref.partition(":")
    if mode not in ("full", "prefix", "other") or (mode == "prefix") != bool(arg):
        raise SystemExit(f"--spk-ref {args.spk_ref!r}: use full, prefix:<seconds> or other")
    partners = reference_partners([w[0] for w in wavs]) if mode == "other" else {}
    by_key = {rel: (wav, sr) for rel, wav, sr in wavs}
    mel_rate = cfg.audio.sample_rate / cfg.audio.hop_length

    def ref_for(rel, mel):
        if mode == "full":
            return mel
        if mode == "prefix":
            return mel[:, : max(1, int(float(arg) * mel_rate))]
        w, wsr = by_key[partners[rel]]
        return mel_fn(resample(w, wsr, cfg.audio.sample_rate).to(args.device).unsqueeze(0))
    scores = {v: {"mel_l1": [], "mel_l1_band": [], "mel_rmse": [], "energy_corr": []}
              for v in views}
    per_utt: dict[str, dict[str, float]] = {}
    t0 = time.time()
    for i, (rel, wav, sr) in enumerate(wavs):
        x = resample(wav, sr, cfg.audio.sample_rate).to(args.device)
        mel = mel_fn(x.unsqueeze(0))
        mel = mel[:, : (mel.shape[1] // s) * s]
        if mel.shape[1] == 0:
            raise SystemExit(f"{rel}: shorter than one token")
        ref = ref_for(rel, mel)
        for v in views:
            torch.manual_seed(args.seed + i)  # same noise for every view
            if v == "full":
                hyp = model.reconstruct(mel, ref_mel=ref, nfe=args.nfe, shortcut=shortcut)
            else:
                out = model.encode_mel(mel)
                residual = None
                if v == "tokens_prior" and model.bottleneck.residual_up is not None:
                    # Information-free residual through the real channel
                    # (message 0 + channel noise): what a tokens-only LM feeds.
                    residual = torch.zeros(*out.content_indices.shape,
                                           model.bottleneck.residual_dim, device=args.device)
                hyp = model.detokenize(out.content_indices, out.prosody_indices, ref_mel=ref,
                                       residual=residual, nfe=args.nfe, shortcut=shortcut)
            l1, rmse = mel_distance(mel.cpu(), hyp.cpu())
            nb = band_limited_bins(cfg.audio, sr)
            l1b = mel_distance(mel[..., :nb].cpu(), hyp[..., :nb].cpu())[0]
            ec = energy_correlation(mel.cpu(), hyp.cpu())
            scores[v]["mel_l1"].append(l1)
            scores[v]["mel_l1_band"].append(l1b)
            scores[v]["mel_rmse"].append(rmse)
            scores[v]["energy_corr"].append(ec)
            per_utt.setdefault(rel, {}).update(
                {f"{v}/mel_l1": l1, f"{v}/mel_l1_band": l1b, f"{v}/mel_rmse": rmse,
                 f"{v}/energy_corr": ec})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(wavs)} ({time.time() - t0:.0f}s)")

    report = {
        "ckpt": args.ckpt, "step": ckpt.get("step"),
        "sampler": args.sampler, "nfe": args.nfe, "n_utterances": len(wavs),
        "spk_ref": args.spk_ref,
        "bitrate": bitrate(model),
        "note": "pre-vocoder mel; quote with vocoder_ceiling mel_l1_fact_space",
        "views": {v: {k: mean_std(xs) for k, xs in m.items()} for v, m in scores.items()},
        "per_utterance": per_utt,
        "config": {"bottleneck": dataclasses.asdict(cfg.bottleneck),
                   "leakage": dataclasses.asdict(cfg.leakage)},
    }
    text = json.dumps(report, indent=2)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
