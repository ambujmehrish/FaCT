"""Lambda_ctc sweep (RISK_ASSESSMENT R3 / ablation #2) - run BEFORE the pilot.

SiTok reports that the CTC weight is crucial but does not publish a value;
this sweep characterizes it. Each arm trains from the same seed on the
same shards with only ctc.weight changed, then reports the R3 decision
columns: reconstruction loss, CTC loss, greedy byte-CER (is the content
stream grounded?), ctc_infeasible, and content-codebook utilization (is
CTC pressure collapsing or spreading the lattice?).

Pilot-scale (cluster):
    python scripts/sweep_lambda_ctc.py --config configs/fact_smoke.yaml \
        --shards $WORK/shards/pilot --lambdas 0.3,1.0,3.0 \
        --steps 50000 --batch-size 8 --device cuda --out results/lctc
    # re-evaluate the saved checkpoints on held-out shards, no retraining:
    python scripts/sweep_lambda_ctc.py ... --out results/lctc --eval-only \
        --eval-shards $WORK/shards/pilot_dev

Decision rule (R3): pick the largest lambda whose reconstruction loss is
within noise of the smallest lambda's - grounding is free until it isn't.
CER should fall with lambda; recon plateau-while-CTC-drops means lambda
too high (encoder became an ASR front-end).

Verdict-protocol note: sweep numbers at sandbox step counts validate the
TOOLING only; no lambda verdict is valid below pilot scale (>=50K steps).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import sys
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fact.config import FaCTConfig
from fact.data.dataset import ShardGroupedSampler, ShardedSpeech, collate
from fact.eval.grounding import codebook_stats, greedy_byte_cer, token_lengths, valid_indices
from fact.model import FaCT
from fact.train.trainer import Trainer


def apply_corpus_stats(cfg: FaCTConfig, shards: str) -> None:
    stats_path = Path(shards) / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        if "mel_mean" in stats:
            cfg.audio.mel_mean = stats["mel_mean"]
            cfg.audio.mel_std = stats["mel_std"]


def to_device(batch, device: str):
    return dataclasses.replace(batch, **{
        f.name: getattr(batch, f.name).to(device)
        for f in dataclasses.fields(batch)
        if isinstance(getattr(batch, f.name), torch.Tensor)
    })


@torch.no_grad()
def eval_arm(model: FaCT, eval_batches: list, device: str) -> dict:
    """Losses, greedy byte-CER and content-codebook stats on fixed eval batches.

    The RNG is reseeded first so every arm sees the same flow-matching noise
    and timesteps: the eval losses differ only through the model. CER and
    codebook stats use valid (unpadded) tokens only.
    """
    model.eval()
    torch.manual_seed(0)
    sums = {"recon": 0.0, "ctc": 0.0, "ctc_infeasible": 0.0}
    for batch in eval_batches:
        logs = model.training_step(to_device(batch, device))
        for k in sums:
            sums[k] += float(logs[k])
    result = {f"eval_{k}": v / len(eval_batches) for k, v in sums.items()}
    result["greedy_byte_cer"] = greedy_byte_cer(model, eval_batches, device)
    s = model.cfg.encoder.frame_stack
    all_indices = []
    for batch in eval_batches:
        out = model.encode_mel(batch.mel.to(device))
        if out.content_indices is not None:
            lengths = token_lengths(batch, out.content_indices.shape[1], s)
            all_indices.append(valid_indices(out.content_indices, lengths))
    if all_indices:
        stats = codebook_stats(torch.cat(all_indices),
                               model.bottleneck.content_codebook_size)
        result["content_utilization"] = stats["utilization"]
        result["content_perplexity"] = stats["perplexity"]
        result["content_tokens"] = stats["tokens"]
    model.train()
    return result


PARAM = "ctc.weight"  # set from --param in main()


def arm_config(base_cfg: FaCTConfig, lam: float) -> FaCTConfig:
    """Copy of the base config with the swept field (PARAM) set to `lam`."""
    cfg = FaCTConfig.from_dict(copy.deepcopy(base_cfg.to_dict()))
    section, field = PARAM.split(".")
    sub = getattr(cfg, section)
    old = getattr(sub, field)
    setattr(sub, field, type(old)(lam))
    return cfg


def arm_dir(out_dir: Path, lam: float) -> Path:
    # ctc.weight keeps the historical lam_<v> names (existing checkpoints).
    prefix = "lam" if PARAM == "ctc.weight" else PARAM.split(".")[1]
    return out_dir / f"{prefix}_{lam:g}"


def eval_checkpoint(lam: float, base_cfg: FaCTConfig, out_dir: Path, device: str,
                    eval_batches: list) -> dict:
    """Evaluate the arm's saved final checkpoint without retraining."""
    ckpts = sorted(arm_dir(out_dir, lam).glob("step_*.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoint for {PARAM}={lam:g} in {arm_dir(out_dir, lam)}")
    state = torch.load(ckpts[-1], map_location="cpu", weights_only=True)
    model = FaCT(arm_config(base_cfg, lam))
    model.load_state_dict(state["model"])
    model.to(device)
    return {"param": PARAM, "lambda": lam, "steps": state["step"], "checkpoint": str(ckpts[-1]),
            "final": None, **eval_arm(model, eval_batches, device)}


def run_arm(lam: float, base_cfg: FaCTConfig, shards: str, steps: int,
            batch_size: int, device: str, out_dir: Path, seed: int,
            eval_batches: list, num_workers: int = 0) -> dict:
    cfg = arm_config(base_cfg, lam)
    cfg.train.ckpt_every = steps
    cfg.train.log_every = max(1, min(500, steps // 10))

    torch.manual_seed(seed)
    model = FaCT(cfg)
    curve: list[dict] = []
    trainer = Trainer(model, cfg, stage="a", device=device,
                      ckpt_dir=str(arm_dir(out_dir, lam)),
                      log_fn=lambda s: curve.append({"log": s}) or print(s))

    ds = ShardedSpeech(shards, cfg, seed=seed)
    sampler = ShardGroupedSampler(ds, shuffle=True, seed=seed)  # same order across arms
    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                        num_workers=num_workers, collate_fn=partial(collate, cfg=cfg),
                        drop_last=True, pin_memory=torch.cuda.is_available())
    t0 = time.time()
    final_logs = trainer.fit(loader, max_steps=steps, sampler=sampler)
    arm = {
        "param": PARAM,
        "lambda": lam,
        "steps": steps,
        "wall_seconds": round(time.time() - t0, 1),
        "final": final_logs,
        **eval_arm(model, eval_batches, device),
    }
    return arm


def write_table(arms: list[dict], out_dir: Path) -> str:
    cols = ["lambda" if PARAM == "ctc.weight" else PARAM, "steps",
            "eval_recon", "eval_ctc", "greedy_byte_cer",
            "content_utilization", "content_perplexity", "eval_ctc_infeasible"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for a in arms:
        row = [f"{a['lambda']:g}", str(a["steps"]),
               f"{a['eval_recon']:.4f}", f"{a['eval_ctc']:.4f}",
               f"{a['greedy_byte_cer']:.3f}",
               f"{a.get('content_utilization', float('nan')):.3f}",
               f"{a.get('content_perplexity', float('nan')):.1f}",
               f"{a['eval_ctc_infeasible']:.3f}"]
        lines.append("| " + " | ".join(row) + " |")
    text = "\n".join(lines) + "\n"
    (out_dir / "sweep_table.md").write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--lambdas", default="0.3,1.0,3.0",
                    help="comma-separated values of --param")
    ap.add_argument("--param", default="ctc.weight",
                    help="config field to sweep, section.field "
                         "(e.g. bottleneck.residual_capacity)")
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-shards", default=None,
                    help="held-out shard dir (e.g. shards/pilot_dev); default: "
                         "the last --eval-utts training utterances")
    ap.add_argument("--eval-utts", type=int, default=256)
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; evaluate each arm's saved checkpoint")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE",
                    help="extra fixed override for every arm (repeatable), e.g. to "
                         "keep an arm matched to an older sweep")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    global PARAM
    PARAM = args.param
    cfg = FaCTConfig.from_yaml(args.config)
    for item in args.set:
        key, _, value = item.partition("=")
        section, field = key.split(".")
        sub = getattr(cfg, section)
        old = getattr(sub, field)  # AttributeError on an unknown field
        setattr(sub, field, value if isinstance(old, str) else type(old)(float(value)))
    arm_config(cfg, 0)  # fail fast on an unknown --param
    apply_corpus_stats(cfg, args.shards)
    args.out.mkdir(parents=True, exist_ok=True)

    # Fixed evaluation batches, identical across arms.
    if args.eval_shards:
        ds = ShardedSpeech(args.eval_shards, cfg)
        idxs = list(range(min(args.eval_utts, len(ds))))
    else:
        ds = ShardedSpeech(args.shards, cfg)
        idxs = list(range(max(0, len(ds) - args.eval_utts), len(ds)))
    eval_batches = [collate([ds[i] for i in idxs[j : j + 4]], cfg)
                    for j in range(0, len(idxs), 4)]

    arms = []
    for lam in [float(x) for x in args.lambdas.split(",")]:
        print(f"\n===== {PARAM} = {lam:g} =====")
        if args.eval_only:
            arms.append(eval_checkpoint(lam, cfg, args.out, args.device, eval_batches))
        else:
            arms.append(run_arm(lam, cfg, args.shards, args.steps, args.batch_size,
                                args.device, args.out, args.seed, eval_batches,
                                num_workers=args.num_workers))
        (args.out / "sweep_report.json").write_text(json.dumps(arms, indent=2))
    print("\n" + write_table(arms, args.out))
    if not args.eval_only and args.steps < 50_000:
        print("NOTE: below pilot scale (50K steps) this validates tooling only; "
              "no lambda verdict (verdict protocol / RISK R3).")


if __name__ == "__main__":
    main()
