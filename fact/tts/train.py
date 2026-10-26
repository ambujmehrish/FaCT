"""Train a TTS LM on one codec's tokens (DDP via torchrun; auto-resume).

    torchrun ... -m fact.tts.train --tokens <data>/tts_tokens/fact \
        --require <data>/tts_tokens/mimi2 --ckpt-dir <runs>/tts_fact --steps 200000

--require: token dirs of the OTHER compared codecs; only utterances present
in all of them are used, so every LM trains on the same data. The same
--require set (all compared codecs) must be given to every LM.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from ..train.trainer import cosine_lr, ddp_env, setup_distributed
from .data import TokenBatches, common_shards, split_shards
from .model import TTSLM, TTSConfig, param_count


def save(path: Path, state: dict, keep_every: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.partial")
    torch.save(state, tmp)
    os.replace(tmp, path)
    ckpts = sorted(path.parent.glob("step_*.pt"))
    for c in ckpts[:-2]:
        if int(c.stem.split("_")[1]) % keep_every:
            c.unlink(missing_ok=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m fact.tts.train")
    ap.add_argument("--tokens", type=Path, required=True)
    ap.add_argument("--require", type=Path, nargs="*", default=[])
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=4_000)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=int, default=12_000, help="positions per GPU batch")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--max-text", type=int, default=400)
    ap.add_argument("--n-val-shards", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=5_000)
    ap.add_argument("--val-batches", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=5_000)
    ap.add_argument("--keep-every", type=int, default=50_000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    meta_p = a.tokens / "meta.json"
    if not meta_p.exists():
        ap.error(f"{meta_p} missing (fact.tts.tokenize)")
    for r in a.require:
        if not (r / "meta.json").exists():
            ap.error(f"{r}/meta.json missing")
    meta = json.loads(meta_p.read_text())
    names = common_shards(a.tokens, a.require)
    if len(names) <= a.n_val_shards:
        ap.error(f"only {len(names)} shards common to all token dirs")
    train_names, val_names = split_shards(names, a.n_val_shards)

    is_ddp, rank, world = ddp_env()
    device = setup_distributed("cuda" if a.device.startswith("cuda") else a.device)
    torch.manual_seed(a.seed)
    cfg = TTSConfig(vocab_sizes=tuple(meta["vocab_sizes"]), dim=a.dim, n_layers=a.layers,
                    n_heads=a.heads, max_frames=a.max_frames, max_text=a.max_text)
    model = TTSLM(cfg).to(device)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))
    run = {"tokens": str(a.tokens), "require": sorted(str(r) for r in a.require), "world": world,
           "steps": a.steps, "lr": a.lr, "max_tokens": a.max_tokens, "n_train_shards": len(train_names),
           "n_val_shards": len(val_names), "codec_meta": meta}
    step = 0
    ckpts = sorted(a.ckpt_dir.glob("step_*.pt"))
    if ckpts:
        ck = torch.load(ckpts[-1], map_location="cpu", weights_only=True)
        if ck["config"] != cfg.to_dict() or ck["run"] != run:
            raise SystemExit(f"refusing to resume {ckpts[-1]}: config/run settings differ")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
    net = WithLoss(model)
    if is_ddp:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
    if rank == 0:
        print(f"TTS LM {param_count(model) / 1e6:.0f}M params | codec {meta['codec']} {meta['vocab_sizes']} "
              f"| {len(train_names)} train / {len(val_names)} val shards | resume step {step}", flush=True)

    # Resumed jobs reshuffle with a step-dependent seed (data order differs, the stream stays infinite).
    ds = TokenBatches(a.tokens, a.require, train_names, cfg, a.max_tokens, rank, world, seed=a.seed + step)
    loader = iter(DataLoader(ds, batch_size=None, num_workers=a.workers, pin_memory=True,
                             persistent_workers=a.workers > 0))

    def to_dev(b):
        return {k: v.to(device, non_blocking=True) for k, v in b.items()}

    @torch.no_grad()
    def validate():
        model.eval()
        vds = TokenBatches(a.tokens, a.require, val_names, cfg, a.max_tokens, rank, world, infinite=False)
        tot, n = {}, 0
        for i, b in enumerate(vds):
            if i >= a.val_batches:
                break
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                _, logs = model.loss(to_dev(b))
            for k, v in logs.items():
                tot[k] = tot.get(k, 0.0) + v
            n += 1
        model.train()
        t = torch.tensor([tot.get(f"ce{s}", 0.0) for s in range(cfg.n_streams)] + [n], device=device,
                         dtype=torch.float64)
        if is_ddp:
            dist.all_reduce(t)
        return [float(t[s] / max(t[-1], 1)) for s in range(cfg.n_streams)]

    t0 = time.time()
    model.train()
    while step < a.steps:
        lr = cosine_lr(step, a.warmup, a.steps, a.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        b = to_dev(next(loader))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            loss, logs = net(b)  # through DDP so gradient hooks fire
        if not math.isfinite(float(loss.detach())):
            raise SystemExit(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if rank == 0 and step % a.log_every == 0:
            ce = " ".join(f"{k}={v:.4f}" for k, v in logs.items())
            print(f"[tts] step {step}/{a.steps} lr={lr:.2e} {ce} ({time.time() - t0:.0f}s)", flush=True)
        if step % a.val_every == 0 or step == a.steps:
            v = validate()
            if rank == 0:
                print(f"[tts] val step {step}: " + " ".join(f"ce{s}={x:.4f}" for s, x in enumerate(v)), flush=True)
        if rank == 0 and (step % a.ckpt_every == 0 or step == a.steps):
            save(a.ckpt_dir / f"step_{step:07d}.pt",
                 {"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                  "config": cfg.to_dict(), "run": run}, a.keep_every)
    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


class WithLoss(torch.nn.Module):
    """forward(batch) = model.loss(batch), for DistributedDataParallel."""

    def __init__(self, model: TTSLM):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model.loss(batch)


if __name__ == "__main__":
    main()
