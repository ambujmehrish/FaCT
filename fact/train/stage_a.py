"""Stage A: train encoder + factorized bottleneck + flow-matching decoder.

Usage:
    python -m fact.train.stage_a --config configs/fact_base.yaml \
        --shards shards/ --ckpt-dir runs/stage_a
    # or a CPU smoke run on synthetic data:
    python -m fact.train.stage_a --synthetic --tiny --steps 30
"""

from __future__ import annotations

import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader

from ..config import FaCTConfig, tiny_config
from ..model import FaCT
from .trainer import Trainer, ddp_env, setup_distributed


def build_loader(args, cfg: FaCTConfig):
    """Returns (loader, sampler); sampler is a DistributedSampler under DDP."""
    if args.synthetic:
        from ..data.synthetic import SyntheticSpeech, collate
        ds = SyntheticSpeech(cfg, n_items=args.batch_size * 8, seconds=args.seconds)
    else:
        from ..data.dataset import ShardedSpeech, collate
        ds = ShardedSpeech(args.shards, cfg, crop_tokens=args.crop_tokens)
    is_ddp, _, _ = ddp_env()
    sampler = None
    if is_ddp:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(ds, shuffle=True, drop_last=True)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=args.num_workers, collate_fn=partial(collate, cfg=cfg),
        drop_last=True, pin_memory=torch.cuda.is_available(),
    )
    return loader, sampler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--tiny", action="store_true", help="tiny CPU config")
    ap.add_argument("--synthetic", action="store_true", help="synthetic smoke data")
    ap.add_argument("--shards", default=None)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--crop-tokens", type=int, default=None,
                    help="fixed token-count crops (the RQ4 knob)")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.tiny:
        cfg = tiny_config()
    elif args.config:
        cfg = FaCTConfig.from_yaml(args.config)
    else:
        cfg = FaCTConfig()
    if not args.synthetic and not args.shards:
        ap.error("Provide --shards or use --synthetic")

    device = setup_distributed("cuda" if args.device.startswith("cuda") else args.device)
    model = FaCT(cfg)
    trainer = Trainer(model, cfg, stage="a", device=device, ckpt_dir=args.ckpt_dir)
    if trainer.rank == 0:
        print(f"Token rate: {cfg.token_rate_hz:.2f} Hz | params: {model.param_counts()}")
    if args.resume:
        trainer.load(args.resume)
    loader, sampler = build_loader(args, cfg)
    logs = trainer.fit(loader, max_steps=args.steps, sampler=sampler)
    if trainer.rank == 0:
        print(f"Final: {logs}")


if __name__ == "__main__":
    main()
