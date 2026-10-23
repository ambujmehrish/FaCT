"""Stage D (#2): deterministic waveform decoder on FROZEN FaCT tokens.

    torchrun ... -m fact.train.stage_d_wave_decoder \
        --fact-ckpt <runs>/main_c8_rd05/step_0400000.pt \
        --manifest <manifests>/train.tsv --dev-manifest <manifests>/dev.tsv \
        --ckpt-dir <runs>/wavedec_main_c8_rd05

Every competing codec decodes its tokens with a GAN-trained waveform
decoder learned on those tokens; FaCT's path (sampled flow-matching mel +
a separately trained vocoder) is the odd one out. This trains that kind
of decoder for FaCT without touching the tokenizer: the FaCT checkpoint is
frozen (encoder, bottleneck, speaker encoder), so tokens, bitrate and all
factorization results are unchanged.

Input per mel frame: the FaCT decoder condition (content + prosody
[+ residual] embeddings, repeated from 12.5 Hz to the 50 Hz mel rate)
concatenated with the global speaker vector. The residual is removed for
half of the training items when the FaCT run was trained with residual
dropout, so one decoder serves both the full and the tokens-only view.
Architecture, discriminators and losses are the stage-C vocoder's
(Vocos-style generator, MPD + MRD, hinge + feature matching + mel L1).
Not causal: the FaCT encoder may be, but this decoder sees +-few frames.

Segments are 127 x 480 samples (2.54 s): 128 mel frames = 32 tokens, so
the FaCT encoder sees a whole number of tokens and the decoder output is
exactly the segment length.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import FaCTConfig
from ..model import FaCT
from ..modules.mel import LogMelSpectrogram
from ..vocoder.data import RandomSegments, load_dev_utterances
from ..vocoder.model import (Generator, MelLoss, VocoderConfig, build_discriminators,
                             discriminator_loss, feature_matching_loss, generator_adv_loss,
                             split)
from .stage_c_vocoder import _finite_grads, check_resume, save
from .trainer import cosine_lr, ddp_env, setup_distributed


def load_fact_frozen(path: str, device: str) -> FaCT:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = FaCT(FaCTConfig.from_dict(ck["config"]))
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def fact_condition(model: FaCT, mel: torch.Tensor, drop_residual: torch.Tensor | None = None,
                   ssl: torch.Tensor | None = None) -> torch.Tensor:
    """mel (B, T, n_mels), T a multiple of frame_stack -> decoder input
    (B, T, cond_dim + spk_dim). drop_residual: (B,) bool, True = tokens only."""
    with torch.no_grad():
        out = model.encode_mel(mel, ssl)
        if drop_residual is not None and out.residual_emb is not None:
            keep = (~drop_residual).to(out.residual_emb.dtype)[:, None, None]
            out = dataclasses.replace(out, residual_emb=out.residual_emb * keep)
        cond = out.decoder_condition()  # (B, T_tok, cond_dim)
        cond = cond.repeat_interleave(model.cfg.encoder.frame_stack, dim=1)[:, : mel.shape[1]]
        spk = model.speaker_encoder(model._norm_mel(mel))  # (B, spk_dim)
        return torch.cat([cond, spk[:, None].expand(-1, cond.shape[1], -1)], dim=-1)


def input_dim(model: FaCT) -> int:
    return model.decoder.cond_proj.in_features + model.decoder.spk_proj.in_features


def load_wave_decoder(path: str, device: str = "cpu") -> tuple[Generator, dict]:
    """(generator, checkpoint meta) from a stage-D checkpoint."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    gen = Generator(VocoderConfig.from_dict(ck["config"]), in_dim=ck["in_dim"])
    gen.load_state_dict(ck["generator"])
    return gen.to(device).eval(), {k: ck[k] for k in ("fact_ckpt", "in_dim", "residual_drop")}


@torch.no_grad()
def validate(gen, fact, mel_fn, dev, device) -> dict:
    from ..eval.reconstruction import mel_distance
    gen.eval()
    s = fact.cfg.encoder.frame_stack
    l1s, pesqs = [], []
    for _, wav in dev:
        y = wav.to(device)[None]
        mel = mel_fn(y)
        mel = mel[:, : mel.shape[1] // s * s]
        y_hat = gen(fact_condition(fact, mel))
        l1s.append(mel_distance(mel.cpu(), mel_fn(y_hat).cpu())[0])
        try:
            from ..eval.metrics_audio import pesq_wb
            pesqs.append(pesq_wb(y[0, : y_hat.shape[1]], y_hat[0], gen.cfg.audio.sample_rate))
        except ImportError:
            pass
    gen.train()
    out = {"val_mel_l1": sum(l1s) / len(l1s)}
    if pesqs:
        out["val_pesq_wb"] = sum(pesqs) / len(pesqs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fact-ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dev-manifest", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--tiny", action="store_true", help="tiny generator/discriminators (tests)")
    ap.add_argument("--set", action="append", default=[], metavar="vocoder.FIELD=VALUE")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    is_ddp, rank, world = ddp_env()
    device = setup_distributed("cuda" if args.device.startswith("cuda") else args.device)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    log = print if rank == 0 else (lambda *a, **k: None)

    fact = load_fact_frozen(args.fact_ckpt, device)
    if fact.cfg.encoder.ssl_model:
        raise SystemExit("stage D for SSL-input FaCT runs is not supported yet (needs audio "
                         "at 16 kHz per segment)")
    s = fact.cfg.encoder.frame_stack
    if args.tiny:
        from ..vocoder.model import tiny_vocoder_config
        vc = dataclasses.replace(tiny_vocoder_config(), audio=fact.cfg.audio, segment_frames=7)
    else:
        vc = VocoderConfig(audio=fact.cfg.audio, segment_frames=127, batch_size=8,
                           max_steps=250_000)
    for item in args.set:
        key, _, value = item.partition("=")
        section, fld = key.split(".")
        if section != "vocoder":
            raise SystemExit(f"--set {item}: only vocoder.FIELD is configurable")
        old = getattr(vc, fld)
        setattr(vc, fld, value if isinstance(old, str) else type(old)(float(value)))
    if (vc.segment_frames + 1) % s:
        raise SystemExit(f"segment_frames + 1 = {vc.segment_frames + 1} must be a multiple of "
                         f"frame_stack {s} (whole tokens, output = segment length)")
    max_steps = args.steps or vc.max_steps
    p_drop = 0.5 if (fact.bottleneck.residual_up is not None
                     and fact.cfg.bottleneck.residual_dropout > 0) else 0.0
    in_dim = input_dim(fact)

    torch.manual_seed(vc.seed)
    gen = Generator(vc, in_dim=in_dim).to(device)
    mpd, mrd = build_discriminators(vc)
    mpd, mrd = mpd.to(device), mrd.to(device)
    mel_fn = LogMelSpectrogram(vc.audio).to(device)
    mel_loss = MelLoss(vc).to(device)
    opt_g = torch.optim.AdamW(gen.parameters(), lr=vc.lr, betas=vc.betas)
    d_params = list(mpd.parameters()) + list(mrd.parameters())
    opt_d = torch.optim.AdamW(d_params, lr=vc.lr, betas=vc.betas)

    run = {"world_size": world, "max_steps": max_steps, "manifest": str(args.manifest),
           "fact_ckpt": str(args.fact_ckpt)}
    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=True)
        check_resume(ck["config"], vc.to_dict(), ck.get("run", {}), run, args.resume)
        gen.load_state_dict(ck["generator"])
        mpd.load_state_dict(ck["mpd"])
        mrd.load_state_dict(ck["mrd"])
        opt_g.load_state_dict(ck["opt_g"])
        opt_d.load_state_dict(ck["opt_d"])
        step = ck["step"]
        log(f"resumed from {args.resume} at step {step}")

    g_net, mpd_net, mrd_net = gen, mpd, mrd
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        ids = [int(os.environ.get("LOCAL_RANK", 0))] if device.startswith("cuda") else None
        g_net, mpd_net, mrd_net = DDP(gen, device_ids=ids), DDP(mpd, device_ids=ids), DDP(mrd, device_ids=ids)

    ds = RandomSegments(args.manifest, vc.audio.sample_rate, vc.segment_samples, vc.seed,
                        rank=rank, step_offset=step, gain_db=vc.gain_db)
    loader = DataLoader(ds, batch_size=vc.batch_size, num_workers=args.num_workers,
                        pin_memory=device.startswith("cuda"),
                        persistent_workers=args.num_workers > 0)
    dev = load_dev_utterances(args.dev_manifest, vc.val_utts, vc.audio.sample_rate) \
        if rank == 0 else None
    log(f"stage D: FaCT {args.fact_ckpt} frozen; decoder input {in_dim}-dim, residual "
        f"dropped for {p_drop:.0%} of items; generator "
        f"{sum(p.numel() for p in gen.parameters()) / 1e6:.1f}M; {world} rank(s) x batch "
        f"{vc.batch_size} x {vc.segment_samples} samples")

    torch.manual_seed(vc.seed + 1000 * rank + step)
    ckpt_dir = Path(args.ckpt_dir)
    skipped = 0
    t0 = time.time()
    it = iter(loader)
    while step < max_steps:
        lr = cosine_lr(step, vc.warmup_steps, max_steps, vc.lr)
        for opt in (opt_g, opt_d):
            for g in opt.param_groups:
                g["lr"] = lr
        y = next(it).to(device, non_blocking=True)
        b = y.shape[0]
        mel = mel_fn(y)  # (B, segment_frames + 1, n_mels): a whole number of tokens
        drop = (torch.rand(b, device=device) < p_drop) if p_drop > 0 else None
        y_hat = g_net(fact_condition(fact, mel, drop))[:, : y.shape[1]]

        both = torch.cat([y, y_hat.detach()])
        s_mp, _ = mpd_net(both)
        s_mr, _ = mrd_net(both)
        loss_d = (discriminator_loss(*split(s_mp, b))
                  + vc.mrd_coef * discriminator_loss(*split(s_mr, b)))
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        ok_d = _finite_grads(d_params)
        if ok_d:
            opt_d.step()

        for p in d_params:
            p.requires_grad_(False)
        both = torch.cat([y, y_hat])
        s_mp, f_mp = mpd(both)
        s_mr, f_mr = mrd(both)
        adv = generator_adv_loss(split(s_mp, b)[1]) + vc.mrd_coef * generator_adv_loss(split(s_mr, b)[1])
        fm = feature_matching_loss(f_mp, b) + vc.mrd_coef * feature_matching_loss(f_mr, b)
        mel_l = mel_loss(y, y_hat)
        loss_g = adv + fm + vc.mel_coef * mel_l
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        for p in d_params:
            p.requires_grad_(True)
        ok_g = _finite_grads(list(gen.parameters()))
        if ok_g:
            opt_g.step()
        if not (ok_d and ok_g):
            skipped += 1
            log(f"[stage D] step {step + 1}: non-finite gradients, update skipped ({skipped})")
        step += 1

        if step % vc.log_every == 0:
            log(f"[stage D] step {step}/{max_steps} lr={lr:.2e} d={loss_d.item():.4f} "
                f"adv={adv.item():.4f} fm={fm.item():.4f} mel={mel_l.item():.4f} "
                f"({time.time() - t0:.0f}s)")
        if rank == 0 and step % vc.val_every == 0:
            log(f"[stage D] step {step} " + " ".join(
                f"{k}={v:.4f}" for k, v in validate(gen, fact, mel_fn, dev, device).items()))
        if rank == 0 and step % vc.ckpt_every == 0:
            save(ckpt_dir / f"step_{step:07d}.pt", {
                "generator": gen.state_dict(), "mpd": mpd.state_dict(), "mrd": mrd.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                "step": step, "config": vc.to_dict(), "run": run,
                "fact_ckpt": str(args.fact_ckpt), "in_dim": in_dim, "residual_drop": p_drop,
            }, vc.keep_every)
    if is_ddp:
        import torch.distributed as dist
        dist.barrier()


if __name__ == "__main__":
    main()
