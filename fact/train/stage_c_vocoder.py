"""Stage C: train the mel vocoder on ground-truth FaCT log-mels.

    torchrun --nproc_per_node=4 -m fact.train.stage_c_vocoder \
        --config configs/fact_base.yaml \
        --manifest <manifests>/train.tsv --dev-manifest <manifests>/dev.tsv \
        --ckpt-dir <runs>/vocoder

The vocoder learns mel -> waveform from real audio (mel computed on the
fly with FaCT's own LogMelSpectrogram), so it is independent of any FaCT
checkpoint and serves every run and view. --config supplies only the mel
space (audio section), which must match the FaCT checkpoints it will
vocode; --set vocoder.FIELD=VALUE overrides vocoder fields.

Resume: --resume <ckpt> (slurm/stage_c_vocoder.sbatch picks the newest).
The vocoder config and run settings must match the checkpoint's.
Validation (rank 0, every val_every steps): copy-synthesis mel-L1 in the
input mel space on the first val_utts dev utterances, plus PESQ-WB.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from ..config import FaCTConfig
from ..modules.mel import LogMelSpectrogram
from ..vocoder.data import RandomSegments, load_dev_utterances
from ..vocoder.model import (Generator, MelLoss, VocoderConfig, build_discriminators,
                             discriminator_loss, feature_matching_loss,
                             generator_adv_loss, split, tiny_vocoder_config)
from .trainer import cosine_lr, ddp_env, setup_distributed

RESUME_MUTABLE = {"log_every", "ckpt_every", "val_every", "val_utts"}


def save(path: Path, state: dict, keep_every: int) -> None:
    """Atomic save, then keep every keep_every-th checkpoint plus the newest two."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    if keep_every > 0:
        for c in sorted(path.parent.glob("step_*.pt"))[:-2]:
            if int(c.stem.split("_")[1]) % keep_every:
                c.unlink(missing_ok=True)


def check_resume(saved: dict, now: dict, saved_run: dict, now_run: dict, path: str) -> None:
    diff = [f"{k}: checkpoint={saved.get(k)!r} now={now.get(k)!r}"
            for k in sorted(set(saved) | set(now))
            if k not in RESUME_MUTABLE and _norm(saved.get(k)) != _norm(now.get(k))]
    diff += [f"run.{k}: checkpoint={v!r} now={now_run.get(k)!r}"
             for k, v in saved_run.items() if now_run.get(k) != v]
    if diff:
        raise SystemExit(f"Refusing to resume {path}: settings differ from the run that "
                         "wrote it:\n  " + "\n  ".join(diff))


def _norm(v):
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


@torch.no_grad()
def validate(gen: Generator, mel_fn: LogMelSpectrogram, dev, device: str) -> dict:
    from ..eval.reconstruction import mel_distance
    gen.eval()
    l1s, pesqs = [], []
    for _, wav in dev:
        y = wav.to(device).unsqueeze(0)
        mel = mel_fn(y)
        y_hat = gen(mel)
        l1s.append(mel_distance(mel.cpu(), mel_fn(y_hat).cpu())[0])
        try:
            from ..eval.metrics_audio import pesq_wb
            pesqs.append(pesq_wb(y[0], y_hat[0], gen.cfg.audio.sample_rate))
        except ImportError:
            pass
    gen.train()
    out = {"val_mel_l1": sum(l1s) / len(l1s)}
    if pesqs:
        out["val_pesq_wb"] = sum(pesqs) / len(pesqs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="FaCT config; only its audio section is used")
    ap.add_argument("--tiny", action="store_true", help="tiny CPU config (tests)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dev-manifest", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="vocoder.FIELD=VALUE")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.tiny:
        vc = tiny_vocoder_config()
    else:
        vc = VocoderConfig(audio=FaCTConfig.from_yaml(args.config).audio if args.config
                           else FaCTConfig().audio)
    for item in args.set:
        key, _, value = item.partition("=")
        section, fld = key.split(".")
        if section != "vocoder":
            raise SystemExit(f"--set {item}: only vocoder.FIELD is configurable here")
        old = getattr(vc, fld)  # AttributeError on an unknown field
        setattr(vc, fld, value if isinstance(old, str) else type(old)(float(value)))
    max_steps = args.steps or vc.max_steps

    is_ddp, rank, world = ddp_env()
    device = setup_distributed("cuda" if args.device.startswith("cuda") else args.device)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    log = print if rank == 0 else (lambda *a, **k: None)

    torch.manual_seed(vc.seed)  # identical initial weights on every rank
    gen = Generator(vc).to(device)
    mpd, mrd = build_discriminators(vc)
    mpd, mrd = mpd.to(device), mrd.to(device)
    mel_fn = LogMelSpectrogram(vc.audio).to(device)
    mel_loss = MelLoss(vc).to(device)
    opt_g = torch.optim.AdamW(gen.parameters(), lr=vc.lr, betas=vc.betas)
    opt_d = torch.optim.AdamW(list(mpd.parameters()) + list(mrd.parameters()),
                              lr=vc.lr, betas=vc.betas)

    run = {"world_size": world, "max_steps": max_steps, "manifest": str(args.manifest)}
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
    log(f"vocoder: generator {sum(p.numel() for p in gen.parameters()) / 1e6:.1f}M params, "
        f"MPD+MRD {sum(p.numel() for p in [*mpd.parameters(), *mrd.parameters()]) / 1e6:.1f}M; "
        f"iSTFT n_fft={vc.n_fft} hop={vc.audio.hop_length}; {world} rank(s) x batch "
        f"{vc.batch_size} x {vc.segment_samples} samples")

    d_params = list(mpd.parameters()) + list(mrd.parameters())
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
        y_hat = g_net(mel_fn(y))
        y_hat = y_hat[:, : y.shape[1]]

        # Discriminator step: real and fake in one forward per discriminator.
        # DDP all-reduces gradients, so the finite check agrees on every rank.
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

        # Generator step through the unwrapped, frozen discriminators: no D
        # gradients are computed and nothing of D is all-reduced here.
        for p in d_params:
            p.requires_grad_(False)
        both = torch.cat([y, y_hat])
        s_mp, f_mp = mpd(both)
        s_mr, f_mr = mrd(both)
        adv = generator_adv_loss(split(s_mp, b)[1]) + vc.mrd_coef * generator_adv_loss(split(s_mr, b)[1])
        fm = feature_matching_loss(f_mp, b) + vc.mrd_coef * feature_matching_loss(f_mr, b)
        mel = mel_loss(y, y_hat)
        loss_g = adv + fm + vc.mel_coef * mel
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        for p in d_params:
            p.requires_grad_(True)
        ok_g = _finite_grads(list(gen.parameters()))
        if ok_g:
            opt_g.step()
        if not (ok_d and ok_g):
            skipped += 1
            log(f"[vocoder] step {step + 1}: non-finite gradients "
                f"({'D' if not ok_d else ''}{'G' if not ok_g else ''}), update skipped ({skipped})")
        step += 1

        if step % vc.log_every == 0:
            log(f"[vocoder] step {step}/{max_steps} lr={lr:.2e} d={loss_d.item():.4f} "
                f"adv={adv.item():.4f} fm={fm.item():.4f} mel={mel.item():.4f} "
                f"({time.time() - t0:.0f}s)")
        if rank == 0 and step % vc.val_every == 0:
            log(f"[vocoder] step {step} " + " ".join(
                f"{k}={v:.4f}" for k, v in validate(gen, mel_fn, dev, device).items()))
        if rank == 0 and step % vc.ckpt_every == 0:
            save(ckpt_dir / f"step_{step:07d}.pt", {
                "generator": gen.state_dict(), "mpd": mpd.state_dict(), "mrd": mrd.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                "step": step, "config": vc.to_dict(), "run": run,
            }, vc.keep_every)
    if is_ddp:
        dist.barrier()
    if rank == 0 and torch.cuda.is_available():
        log(f"Peak GPU memory (rank 0): {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


def _finite_grads(params: list) -> bool:
    return all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)


def load_generator(path: str, device: str = "cpu") -> Generator:
    """Generator from a stage-C checkpoint (config read from the checkpoint)."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    gen = Generator(VocoderConfig.from_dict(ck["config"]))
    gen.load_state_dict(ck["generator"])
    return gen.to(device).eval()


if __name__ == "__main__":
    main()
