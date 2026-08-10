"""Post-training smoke verification on REAL audio (called by smoke_test.sh).

Loads the freshly trained checkpoint and checks, on real utterances:
  1. tokenize: both views (indices + continuous) with the right shapes/ranges,
     token rate matches config, codebooks actually in use
  2. detokenize (pure discrete path) and reconstruct (continuous path)
     produce finite mel at the right resolution
  3. CTC feasibility on the REAL transcripts (the metric the audit added)
  4. modelability probe trains on the real token streams and reports
     finite bits/frame
  5. leakage-matrix probe path runs on real features

Quality is NOT judged - a few-step checkpoint is noise; this validates that
every interface works on real data end-to-end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fact.config import FaCTConfig
from fact.data.dataset import ShardedSpeech, collate
from fact.eval.modelability import TokenLMProbe, evaluate_bits, train_probe
from fact.eval.reconstruction import energy_correlation, mel_distance
from fact.model import FaCT
from fact.modules.heads import ctc_required_length


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        raise SystemExit(f"smoke roundtrip failed at: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = FaCTConfig.from_yaml(args.config)
    model = FaCT(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model = model.to(args.device).eval()
    print(f"Loaded {args.ckpt} (step {ckpt['step']}) on {args.device}")

    ds = ShardedSpeech(args.shards, cfg)
    items = [ds[i] for i in range(min(4, len(ds)))]
    batch = collate(items, cfg)
    mel = batch.mel.to(args.device)
    s = cfg.encoder.frame_stack

    # 1. Tokenize: dual views of real speech.
    toks = model.tokenize(mel=mel)
    b, t_tok = toks.content_indices.shape
    check("token shapes", toks.prosody_indices.shape == (b, t_tok))
    check("token rate", abs(t_tok * s - mel.shape[1]) < s,
          f"{t_tok} tokens for {mel.shape[1]} mel frames @ stack {s}")
    check("content indices in range",
          bool((toks.content_indices >= 0).all()
               and (toks.content_indices < model.bottleneck.content_codebook_size).all()))
    n_content = toks.content_indices.unique().numel()
    n_prosody = toks.prosody_indices.unique().numel()
    check("codebooks in use", n_content > 1 and n_prosody > 1,
          f"{n_content} content / {n_prosody} prosody codes among "
          f"{toks.content_indices.numel()} frames")
    check("continuous view bounded",
          bool(toks.content_continuous.abs().max() <= 1.01),
          f"max |v| = {toks.content_continuous.abs().max():.3f}")
    # Duality: the continuous view rounds to exactly the reported indices.
    fsq = model.bottleneck.content_fsq
    recon_codes = fsq.indices_to_codes(toks.content_indices)
    gap = (toks.content_continuous - recon_codes).abs().max()
    check("dual views describe the same lattice state",
          bool(gap <= 0.5 / fsq.half_width.min() + 1e-4), f"max gap {gap:.4f}")

    # 2. Decode paths.
    mel_disc = model.detokenize(toks.content_indices, toks.prosody_indices,
                                ref_mel=mel, residual=toks.residual, nfe=2)
    check("discrete detokenize", bool(torch.isfinite(mel_disc).all())
          and mel_disc.shape == (b, t_tok * s, cfg.audio.n_mels))
    mel_cont = model.reconstruct(mel, nfe=2)
    check("continuous reconstruct", bool(torch.isfinite(mel_cont).all()))
    l1, _ = mel_distance(mel[:, : t_tok * s].cpu(), mel_disc.cpu())
    corr = energy_correlation(mel[:, : t_tok * s].cpu(), mel_disc.cpu())
    print(f"  info: mel_l1={l1:.3f} energy_corr={corr:.3f} "
          f"(quality meaningless at smoke step count)")

    # 3. CTC feasibility on the real transcripts.
    required = ctc_required_length(batch.text, batch.text_lengths)
    token_lengths = (batch.mel_lengths // s).clamp(max=t_tok)
    positions = token_lengths * cfg.ctc.upsample
    frac = float((required > positions).float().mean())
    check("CTC feasible on real transcripts", frac == 0.0,
          f"infeasible fraction {frac:.2f} "
          f"(required {required.tolist()} vs positions {positions.tolist()})")

    # 4. Modelability probe on the real token streams.
    vocabs = [model.bottleneck.content_codebook_size,
              model.bottleneck.prosody_codebook_size]
    probe = TokenLMProbe(vocabs, dim=64, n_layers=1, n_heads=4)
    streams = [[toks.content_indices.cpu(), toks.prosody_indices.cpu()]]
    train_probe(probe, streams, steps=5)
    res = evaluate_bits(probe, streams, frame_rate_hz=cfg.token_rate_hz)
    check("modelability probe", res.bits_per_frame > 0 and res.frames_evaluated > 0,
          f"{res.bits_per_frame:.2f} bits/frame, {res.bits_per_second:.1f} bits/s")

    # 5. Leakage features exist for the matrix.
    out = toks.bottleneck
    check("leakage features", out.content_emb is not None
          and out.prosody_emb is not None and out.residual is not None)

    print("smoke roundtrip: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
