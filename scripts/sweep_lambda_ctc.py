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
import json
import sys
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fact.config import FaCTConfig
from fact.data.dataset import ShardedSpeech, collate
from fact.eval.wer import word_edit_distance
from fact.model import FaCT
from fact.train.trainer import Trainer


def apply_corpus_stats(cfg: FaCTConfig, shards: str) -> None:
    stats_path = Path(shards) / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        if "mel_mean" in stats:
            cfg.audio.mel_mean = stats["mel_mean"]
            cfg.audio.mel_std = stats["mel_std"]


@torch.no_grad()
def eval_arm(model: FaCT, eval_batches: list) -> dict:
    """Greedy byte-CER + content-codebook stats on held-out batches."""
    model.eval()
    total_dist = total_ref = 0
    all_indices = []
    for batch in eval_batches:
        out = model.encode_mel(batch.mel)
        hyps = model.ctc_head.greedy_decode(out.primary_emb())
        for i, hyp in enumerate(hyps):
            n = int(batch.text_lengths[i])
            if n == 0:
                continue
            ref = batch.text[i, :n].tolist()
            total_dist += word_edit_distance(ref, hyp)
            total_ref += n
        if out.content_indices is not None:
            all_indices.append(out.content_indices.flatten())
    result = {"greedy_byte_cer": total_dist / max(1, total_ref)}
    if all_indices:
        idx = torch.cat(all_indices)
        size = model.bottleneck.content_codebook_size
        counts = torch.bincount(idx, minlength=size).float()
        result["content_utilization"] = float((counts > 0).float().mean())
        probs = counts / counts.sum().clamp_min(1)
        nz = probs[probs > 0]
        result["content_perplexity"] = float(torch.exp(-(nz * nz.log()).sum()))
    model.train()
    return result


def run_arm(lam: float, base_cfg: FaCTConfig, shards: str, steps: int,
            batch_size: int, device: str, out_dir: Path, seed: int,
            eval_batches: list) -> dict:
    cfg = FaCTConfig.from_dict(copy.deepcopy(base_cfg.to_dict()))
    cfg.ctc.weight = lam
    cfg.train.ckpt_every = steps
    cfg.train.log_every = max(1, steps // 10)

    torch.manual_seed(seed)
    model = FaCT(cfg)
    curve: list[dict] = []
    trainer = Trainer(model, cfg, stage="a", device=device,
                      ckpt_dir=str(out_dir / f"lam_{lam:g}"),
                      log_fn=lambda s: curve.append({"log": s}) or print(s))

    ds = ShardedSpeech(shards, cfg, seed=seed)
    g = torch.Generator().manual_seed(seed)  # same batch order across arms
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g,
                        collate_fn=partial(collate, cfg=cfg), drop_last=True)
    t0 = time.time()
    final_logs = trainer.fit(loader, max_steps=steps)
    arm = {
        "lambda": lam,
        "steps": steps,
        "wall_seconds": round(time.time() - t0, 1),
        "final": final_logs,
        **eval_arm(model, eval_batches),
    }
    return arm


def write_table(arms: list[dict], out_dir: Path) -> str:
    cols = ["lambda", "recon", "ctc", "ctc_infeasible", "greedy_byte_cer",
            "content_utilization", "content_perplexity"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for a in arms:
        row = [f"{a['lambda']:g}",
               f"{a['final']['recon']:.4f}", f"{a['final']['ctc']:.4f}",
               f"{a['final']['ctc_infeasible']:.3f}",
               f"{a.get('greedy_byte_cer', float('nan')):.3f}",
               f"{a.get('content_utilization', float('nan')):.3f}",
               f"{a.get('content_perplexity', float('nan')):.1f}"]
        lines.append("| " + " | ".join(row) + " |")
    text = "\n".join(lines) + "\n"
    (out_dir / "sweep_table.md").write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--lambdas", default="0.3,1.0,3.0")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-utts", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = FaCTConfig.from_yaml(args.config)
    apply_corpus_stats(cfg, args.shards)
    args.out.mkdir(parents=True, exist_ok=True)

    # Fixed held-out batches, identical across arms (last utterances).
    ds = ShardedSpeech(args.shards, cfg)
    idxs = list(range(max(0, len(ds) - args.eval_utts), len(ds)))
    eval_batches = [collate([ds[i] for i in idxs[j : j + 4]], cfg)
                    for j in range(0, len(idxs), 4)]

    arms = []
    for lam in [float(x) for x in args.lambdas.split(",")]:
        print(f"\n===== lambda_ctc = {lam:g} =====")
        arms.append(run_arm(lam, cfg, args.shards, args.steps, args.batch_size,
                            args.device, args.out, args.seed, eval_batches))
        (args.out / "sweep_report.json").write_text(json.dumps(arms, indent=2))
    print("\n" + write_table(arms, args.out))
    if args.steps < 50_000:
        print("NOTE: below pilot scale (50K steps) this validates tooling only; "
              "no lambda verdict (verdict protocol / RISK R3).")


if __name__ == "__main__":
    main()
