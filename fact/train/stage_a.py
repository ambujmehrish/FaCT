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
from .trainer import Trainer


def build_loader(args, cfg: FaCTConfig) -> DataLoader:
    if args.synthetic:
        from ..data.synthetic import SyntheticSpeech, collate
        ds = SyntheticSpeech(cfg, n_items=args.batch_size * 8, seconds=args.seconds)
    else:
        from ..data.dataset import ShardedSpeech, collate
        ds = ShardedSpeech(args.shards, cfg, crop_tokens=args.crop_tokens)
    return DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=partial(collate, cfg=cfg),
        drop_last=True,
    )


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

    model = FaCT(cfg)
    print(f"Token rate: {cfg.token_rate_hz:.2f} Hz | params: {model.param_counts()}")
    trainer = Trainer(model, cfg, stage="a", device=args.device, ckpt_dir=args.ckpt_dir)
    if args.resume:
        trainer.load(args.resume)
    logs = trainer.fit(build_loader(args, cfg), max_steps=args.steps)
    print(f"Final: {logs}")


if __name__ == "__main__":
    main()
