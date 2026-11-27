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


def config_diff(a: dict, b: dict, prefix: str = "") -> list[str]:
    """Dotted keys whose values differ between two config dicts."""
    out = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out += config_diff(va, vb, f"{prefix}{k}.")
        elif (list(va) if isinstance(va, (list, tuple)) else va) != (list(vb) if isinstance(vb, (list, tuple)) else vb):
            out.append(f"{prefix}{k}")
    return out


LAYOUT_KEYS = ("world_size", "batch_size", "grad_accum")


def shard_mel_stats(shards: str) -> tuple[float, float]:
    """(mel mean, mel std) of a finished shard directory (its stats.json)."""
    import json
    from pathlib import Path
    p = Path(shards) / "stats.json"
    if not p.exists():
        raise SystemExit(f"{p} not found: finish preprocessing first")
    st = json.loads(p.read_text())
    if st.get("partial") or "mel_mean" not in st or "mel_std" not in st:
        raise SystemExit(f"{p} has no final mel statistics (partial preprocessing)")
    return float(st["mel_mean"]), float(st["mel_std"])


def check_match_run(ref_run: dict, run_meta: dict, same_global_batch: bool = False,
                    may_differ: tuple = (), **fit_meta) -> None:
    """Every run setting the reference recorded must equal this run's
    (fit_meta: the values fit() will record). A setting this run cannot
    compare is an error, never skipped.

    may_differ: run settings that are the declared change of this run (e.g.
    "shards" for the same model on other data); they must exist on both sides.
    same_global_batch: the GPU layout may differ (e.g. 1 node instead of 2)
    as long as world_size x batch_size x grad_accum - the samples per
    optimizer step - is the same; every other setting must still be equal."""
    now = {**run_meta, **fit_meta}
    unknown = [k for k in may_differ if k not in ref_run or k not in now]
    if unknown:
        raise SystemExit(f"--allow-run-change: {unknown} not recorded in the reference run or not set now")
    skip = tuple(may_differ)
    if same_global_batch:
        def samples(d):
            return [d.get(k) for k in LAYOUT_KEYS]
        a, b = samples(ref_run), samples(now)
        if None in a or None in b or a[0] * a[1] * a[2] != b[0] * b[1] * b[2]:
            raise SystemExit(f"--same-global-batch: samples per optimizer step differ "
                             f"({dict(zip(LAYOUT_KEYS, a))} in the reference, {dict(zip(LAYOUT_KEYS, b))} now)")
        skip += LAYOUT_KEYS
    diff = [f"{k}: reference={v!r} now={now.get(k, '<not set>')!r}"
            for k, v in sorted(ref_run.items()) if k not in skip and now.get(k, object()) != v]
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
    ap.add_argument("--ablation-of", default=None, metavar="STEP.PT",
                    help="train an ablation of that run: its exact config, batch and run settings, "
                         "changed only by --set bottleneck.* overrides (any other difference is refused)")
    ap.add_argument("--allow-change", action="append", default=None, metavar="SECTION.FIELD",
                    help="with --ablation-of: config key (or key prefix) that may differ from the "
                         "reference; repeatable; default 'bottleneck.'")
    ap.add_argument("--allow-run-change", action="append", default=None, metavar="KEY",
                    choices=["shards", "audio_index"],
                    help="with --ablation-of: a run setting that is this run's declared change "
                         "(the same model on other data); repeatable")
    ap.add_argument("--mel-stats-from-shards", action="store_true",
                    help="with --ablation-of --allow-run-change shards: take audio.mel_mean/mel_std "
                         "from the shards' stats.json (the corpus statistics, as the reference "
                         "does for its own corpus)")
    ap.add_argument("--same-global-batch", action="store_true",
                    help="with --ablation-of: another GPU layout is accepted when world size x "
                         "per-GPU batch x grad accumulation equals the reference's (give "
                         "--batch-size and/or --set train.grad_accum=N with --allow-change train.grad_accum)")
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
    ref_cfg_dict = None
    if args.ablation_of:
        if args.match_run or args.tiny or args.config or args.init_from:
            ap.error("--ablation-of takes the reference's config: no --match-run/--config/--tiny/--init-from")
        scope = tuple(args.allow_change or ["bottleneck."])
        run_change = tuple(args.allow_run_change or ())
        if args.mel_stats_from_shards and "shards" not in run_change:
            ap.error("--mel-stats-from-shards needs --allow-run-change shards")
        if not all(x.startswith(scope) for x in args.set) or not (args.set or run_change):
            ap.error(f"--ablation-of needs --set overrides, and only of {list(scope)} "
                     f"(or a declared --allow-run-change)")
        if args.mel_stats_from_shards:
            scope += ("audio.mel_mean", "audio.mel_std")
        ref = torch.load(args.ablation_of, map_location="cpu", weights_only=True)
        ref_cfg_dict, ref_run = ref["config"], dict(ref.get("run") or {})
        if not ref_run:
            raise SystemExit(f"--ablation-of {args.ablation_of}: the checkpoint records no run settings")
        cfg = FaCTConfig.from_dict(ref_cfg_dict)
        del ref
        if args.batch_size is None:
            args.batch_size = ref_run.get("batch_size")
        print(f"ablation of {args.ablation_of}: {ref_run}")
    elif args.allow_change or args.same_global_batch or args.allow_run_change or args.mel_stats_from_shards:
        ap.error("--allow-change / --same-global-batch / --allow-run-change / --mel-stats-from-shards "
                 "only apply with --ablation-of")
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
    elif args.match_run or args.ablation_of:
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
    if args.synthetic and args.mel_stats_from_shards:
        ap.error("--mel-stats-from-shards needs real shards")
    if args.synthetic:
        print("*** SYNTHETIC SMOKE DATA - pipeline validation only, "
              "NOT a real experiment ***")
    else:
        from ..data.dataset import verify_mel_stats
        if args.mel_stats_from_shards:
            cfg.audio.mel_mean, cfg.audio.mel_std = shard_mel_stats(args.shards)
            print(f"mel normalization from {args.shards}: mean {cfg.audio.mel_mean} std {cfg.audio.mel_std}")
        verify_mel_stats(args.shards, cfg, allow_mismatch=args.allow_stats_mismatch)

    for item in args.set:
        key, _, value = item.partition("=")
        section, field = key.split(".")
        sub = getattr(cfg, section)
        old = getattr(sub, field)  # AttributeError on an unknown field
        setattr(sub, field, value if isinstance(old, str) else type(old)(float(value)))
    if ref_cfg_dict is not None:
        # Compare in the current schema: a checkpoint written before a config field
        # existed lacks that key, and loading it fills in the default (the behaviour
        # it was trained with). Comparing the raw dict would report those as changes.
        ref_now = FaCTConfig.from_dict(ref_cfg_dict).to_dict()
        added = config_diff(ref_cfg_dict, ref_now)
        if added:
            print(f"reference config predates these fields (defaults used): {added}")
        changed = config_diff(ref_now, cfg.to_dict())
        outside = [k for k in changed if not k.startswith(scope)]
        if outside or not (changed or run_change):
            raise SystemExit(f"--ablation-of: changed {changed or 'nothing'}; only {list(scope)} may differ")
        print(f"ablation changes: {changed}")
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
        if args.ablation_of:
            trainer.run_meta["ablation_of"] = str(args.ablation_of)
            ref_run = {k: v for k, v in ref_run.items() if k not in ("match_run", "ablation_of")}
        else:
            trainer.run_meta["match_run"] = str(args.match_run)
        if args.ablation_of and run_change:
            same = [k for k in run_change if ref_run.get(k) == trainer.run_meta.get(k)]
            if same:
                raise SystemExit(f"--allow-run-change {same}: same value as the reference run")
        check_match_run(ref_run, trainer.run_meta, same_global_batch=args.same_global_batch,
                        may_differ=run_change if args.ablation_of else (),
                        batch_size=args.batch_size,
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
