"""Stage B: shortcut-distill the decoder to 1-2 NFE.

Loads a stage-A checkpoint, freezes encoder + bottleneck + heads (SiTok
recipe), and fine-tunes the decoder with the shortcut self-consistency
objective so 1-2 Euler steps match the many-step flow.

Usage:
    python -m fact.train.stage_b_shortcut --config configs/fact_base.yaml \
        --init runs/stage_a/step_0400000.pt --shards shards/ --ckpt-dir runs/stage_b
"""

from __future__ import annotations

import argparse

import torch

from ..config import FaCTConfig, tiny_config
from ..model import FaCT
from .stage_a import build_loader
from .trainer import Trainer, setup_distributed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--init", default=None, help="stage-A checkpoint")
    ap.add_argument("--shards", default=None)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--crop-tokens", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = tiny_config() if args.tiny else (
        FaCTConfig.from_yaml(args.config) if args.config else FaCTConfig()
    )
    if not args.synthetic and not args.shards:
        ap.error("Provide --shards or use --synthetic")

    device = setup_distributed("cuda" if args.device.startswith("cuda") else args.device)
    model = FaCT(cfg)
    if args.init:
        ckpt = torch.load(args.init, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
    trainer = Trainer(model, cfg, stage="b", device=device, ckpt_dir=args.ckpt_dir)
    loader, sampler = build_loader(args, cfg)
    logs = trainer.fit(loader, max_steps=args.steps, sampler=sampler)
    if trainer.rank == 0:
        print(f"Final: {logs}")


if __name__ == "__main__":
    main()
