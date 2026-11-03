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
    """Returns (loader, sampler). Real shards always use ShardGroupedSampler
    (split across ranks under DDP); synthetic data uses DistributedSampler
    under DDP and plain shuffling otherwise."""
    is_ddp, rank, world = ddp_env()
    sampler = None
    if args.synthetic:
        from ..data.synthetic import SyntheticSpeech, collate
        ds = SyntheticSpeech(cfg, n_items=args.batch_size * 8, seconds=args.seconds)
        if is_ddp:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(ds, shuffle=True, drop_last=True)
    else:
        from ..data.dataset import ShardedSpeech, ShardGroupedSampler, collate
        ds = ShardedSpeech(args.shards, cfg, crop_tokens=args.crop_tokens,
                           audio_index=args.audio_index)
        sampler = ShardGroupedSampler(ds, shuffle=True, seed=cfg.train.seed,
                                      rank=rank, world_size=world)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=args.num_workers, collate_fn=partial(collate, cfg=cfg),
        drop_last=True, pin_memory=torch.cuda.is_available(),
    )
    return loader, sampler


def check_match_run(ref_run: dict, run_meta: dict, **fit_meta) -> None:
    """Every run setting the reference recorded must equal this run's
    (fit_meta: the values fit() will record). A setting this run cannot
    compare is an error, never skipped."""
    now = {**run_meta, **fit_meta}
    diff = [f"{k}: reference={v!r} now={now.get(k, '<not set>')!r}"
            for k, v in sorted(ref_run.items()) if now.get(k, object()) != v]
    if diff:
        raise SystemExit("--match-run: run settings differ from the reference run:\n  "
                         + "\n  ".join(diff))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--tiny", action="store_true", help="tiny CPU config")
    ap.add_argument("--synthetic", action="store_true", help="synthetic smoke data")
    ap.add_argument("--shards", default=None)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init-from", default=None,
                    help="fine-tune: start from this checkpoint's weights (fresh optimizer, "
                         "step 0); ignored when --resume is given (chained jobs)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE",
                    help="config override (repeatable); recorded in every checkpoint")
    ap.add_argument("--ckpt-every", type=int, default=None,
                    help="override train.ckpt_every (short timing/resume tests)")
    ap.add_argument("--batch-size", type=int, default=None, help="per-GPU batch (default 8, or "
                    "--match-run's)")
    ap.add_argument("--match-run", default=None, metavar="STEP.PT",
                    help="retrain another run exactly: take its config and batch size, and refuse "
                         "to start unless every run setting it recorded (world size, shards, audio "
                         "index, steps, grad accumulation, init) is the same here")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--crop-tokens", type=int, default=None,
                    help="fixed token-count crops (the RQ4 knob)")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--audio-index", default=None,
                    help="per-shard audio paths (scripts/build_audio_index.py); required "
                         "for SSL-input configs (encoder.ssl_model)")
    ap.add_argument("--allow-stats-mismatch", action="store_true",
                    help="proceed even if config mel stats differ from shards")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ref_run = None
    if args.match_run:
        if args.tiny or args.config or args.set or args.init_from:
            ap.error("--match-run takes the reference's config: no --config/--set/--tiny/--init-from")
        ref = torch.load(args.match_run, map_location="cpu", weights_only=True)
        cfg, ref_run = FaCTConfig.from_dict(ref["config"]), dict(ref.get("run") or {})
        if not ref_run:
            raise SystemExit(f"--match-run {args.match_run}: the checkpoint records no run settings")
        del ref
        if args.batch_size is None:
            args.batch_size = ref_run.get("batch_size")
        print(f"match-run {args.match_run}: {ref_run}")
    if args.batch_size is None:
        args.batch_size = 8
    if args.tiny:
        cfg = tiny_config()
    elif args.match_run:
        pass
    elif args.config and args.config.endswith(".pt"):
        # Exactly the config another run was trained with (+ --set overrides).
        cfg = FaCTConfig.from_dict(torch.load(args.config, map_location="cpu",
                                              weights_only=True)["config"])
    elif args.config:
        cfg = FaCTConfig.from_yaml(args.config)
    else:
        cfg = FaCTConfig()
    if not args.synthetic and not args.shards:
        ap.error("Provide --shards or use --synthetic")
    if args.synthetic:
        print("*** SYNTHETIC SMOKE DATA - pipeline validation only, "
              "NOT a real experiment ***")
    else:
        from ..data.dataset import verify_mel_stats
        verify_mel_stats(args.shards, cfg, allow_mismatch=args.allow_stats_mismatch)

    for item in args.set:
        key, _, value = item.partition("=")
        section, field = key.split(".")
        sub = getattr(cfg, section)
        old = getattr(sub, field)  # AttributeError on an unknown field
        setattr(sub, field, value if isinstance(old, str) else type(old)(float(value)))
    if args.ckpt_every:
        cfg.train.ckpt_every = args.ckpt_every
    device = setup_distributed("cuda" if args.device.startswith("cuda") else args.device)
    # Same initial weights in every run with the same seed (candidate arms
    # must differ only in their --set overrides); DDP then broadcasts rank 0.
    torch.manual_seed(cfg.train.seed)
    model = FaCT(cfg)
    trainer = Trainer(model, cfg, stage="a", device=device, ckpt_dir=args.ckpt_dir)
    if trainer.rank == 0:
        print(f"Token rate: {cfg.token_rate_hz:.2f} Hz | params: {model.param_counts()}")
    trainer.run_meta["shards"] = str(args.shards) if args.shards else "synthetic"
    if cfg.encoder.ssl_model or cfg.distill.weight > 0:
        if args.synthetic:
            raise SystemExit("SSL input / distillation needs real audio")
        if not args.audio_index:  # default: the index next to the shards
            from pathlib import Path
            args.audio_index = str(Path(args.shards) / "audio_index")
            if not Path(args.audio_index).is_dir():
                raise SystemExit(f"{args.audio_index} missing: run scripts/build_audio_index.py")
            print(f"audio index: {args.audio_index}")
        trainer.run_meta["audio_index"] = str(args.audio_index)
    if args.init_from:  # recorded in every checkpoint; a resumed job must pass the same
        trainer.run_meta["init_from"] = str(args.init_from)
    if ref_run is not None:
        trainer.run_meta["match_run"] = str(args.match_run)
        check_match_run(ref_run, trainer.run_meta, batch_size=args.batch_size,
                        max_steps=args.steps or cfg.train.max_steps,
                        grad_accum=max(1, cfg.train.grad_accum))
    if args.resume:
        trainer.load(args.resume)
    elif args.init_from:
        new = trainer.init_from(args.init_from)
        if trainer.rank == 0:
            print(f"init from {args.init_from}; newly initialised: {sorted(set(k.split('.')[0] for k in new)) or 'none'}")
    loader, sampler = build_loader(args, cfg)
    logs = trainer.fit(loader, max_steps=args.steps, sampler=sampler)
    if trainer.rank == 0:
        print(f"Final: {logs}")
        if torch.cuda.is_available():
            print(f"Peak GPU memory (rank 0): "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
