"""Trainer for stages A and B: single-GPU, or multi-GPU via torchrun DDP.

Launch (CINECA: 4x A100-64GB per node):
    torchrun --nproc_per_node=4 -m fact.train.stage_a --config ... --shards ...

DDP is activated automatically when torchrun's env vars are present. Only
rank 0 logs and checkpoints. Gradient accumulation (train.grad_accum)
recovers the 8-GPU effective batch on 4-GPU nodes.
Stage A trains everything with the full loss; stage B freezes the tokenizer
and shortcut-distills the decoder (SiTok recipe).
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
import torch.distributed as dist

from ..config import FaCTConfig
from ..model import Batch, FaCT


def ddp_env() -> tuple[bool, int, int]:
    """(is_distributed, rank, world_size) from torchrun env."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return True, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    return False, 0, 1


def setup_distributed(device_type: str = "cuda") -> str:
    """Initialize the process group; returns this rank's device string."""
    is_ddp, rank, _ = ddp_env()
    if not is_ddp:
        return device_type if device_type != "cuda" else "cuda:0"
    backend = "nccl" if device_type == "cuda" else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
        return f"cuda:{local_rank}"
    return device_type


def cosine_lr(step: int, warmup: int, max_steps: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


@dataclass
class TrainState:
    step: int = 0
    best_loss: float = float("inf")
    # Data position, so a resumed run continues the epoch instead of
    # replaying it from the start (chained 24 h jobs would otherwise keep
    # re-reading the first part of the corpus).
    epoch: int = 0
    batches_in_epoch: int = 0


class Trainer:
    def __init__(self, model: FaCT, cfg: FaCTConfig, stage: str = "a",
                 device: str = "cpu", ckpt_dir: Optional[str] = None,
                 log_fn: Callable[[str], None] = print):
        self.is_ddp, self.rank, self.world_size = ddp_env()
        if str(device).startswith("cuda"):
            # PyTorch defaults to strict fp32 matmuls; TF32 is several times
            # faster on A100 at negligible accuracy cost for this model.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.model = model.to(device)
        self.cfg = cfg
        self.stage = stage
        self.device = device
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.log_fn = log_fn if self.rank == 0 else (lambda s: None)
        self.state = TrainState()
        # Run settings outside the config that fix the data position and the
        # effective batch; saved in checkpoints and checked on resume.
        self.run_meta: dict = {"world_size": self.world_size}
        self._resumed_meta: Optional[dict] = None
        self.skipped_steps = 0

        if stage == "b":
            self.model.freeze_tokenizer()

        self.net: torch.nn.Module = self.model
        if self.is_ddp:
            from torch.nn.parallel import DistributedDataParallel
            self.net = DistributedDataParallel(
                self.model,
                device_ids=[int(os.environ.get("LOCAL_RANK", 0))]
                if device.startswith("cuda") else None,
                # Stage B freezes most params; GRL probes can also leave
                # branches unused on some variants.
                find_unused_parameters=(stage == "b"),
            )

        params = [p for p in self.model.parameters() if p.requires_grad]
        tc = cfg.train
        self.opt = torch.optim.AdamW(
            params, lr=tc.lr, betas=tc.betas, weight_decay=tc.weight_decay
        )

    def _to_device(self, batch: Batch) -> Batch:
        def mv(x):
            return x.to(self.device, non_blocking=True) if isinstance(x, torch.Tensor) else x
        return Batch(
            mel=mv(batch.mel), text=mv(batch.text), text_lengths=mv(batch.text_lengths),
            f0_bins=mv(batch.f0_bins), energy=mv(batch.energy),
            ref_mel=mv(batch.ref_mel), mel_lengths=mv(batch.mel_lengths),
        )

    def fit(self, loader: Iterable[Batch], max_steps: Optional[int] = None,
            sampler=None) -> dict[str, float]:
        tc = self.cfg.train
        max_steps = max_steps or tc.max_steps
        accum = max(1, tc.grad_accum)
        self.net.train()
        last_logs: dict[str, float] = {}
        t0 = time.time()

        self.run_meta.update({"max_steps": max_steps, "grad_accum": accum,
                              "batch_size": getattr(loader, "batch_size", None)})
        if self._resumed_meta:
            diff = [f"{k}: checkpoint={v!r} now={self.run_meta.get(k)!r}"
                    for k, v in self._resumed_meta.items() if self.run_meta.get(k) != v]
            if diff:
                raise SystemExit("Refusing to resume: run settings differ from the run "
                                 "that wrote the checkpoint (data position / effective "
                                 "batch / LR schedule would change):\n  " + "\n  ".join(diff))
        # Training-phase RNG: distinct per rank (independent noise/dropout masks
        # across GPUs), reproducible, and advanced on resume.
        torch.manual_seed(self.cfg.train.seed + 1000 * self.rank + self.state.step)
        if sampler is not None:
            if self.state.step > 0 and hasattr(sampler, "set_skip"):
                # Every rank consumes exactly step * accum micro-batches, so each
                # rank recomputes ITS OWN position from its own epoch lengths
                # (ranks hold different shards, so positions differ per rank).
                self.state.epoch, self.state.batches_in_epoch = self._data_position(
                    sampler, loader, self.state.step * accum)
            sampler.set_epoch(self.state.epoch)
            if self.state.batches_in_epoch and hasattr(sampler, "set_skip"):
                sampler.set_skip(self.state.batches_in_epoch * loader.batch_size)
        data_iter = iter(loader)
        while self.state.step < max_steps:
            lr = cosine_lr(self.state.step, tc.warmup_steps, max_steps, tc.lr)
            for g in self.opt.param_groups:
                g["lr"] = lr

            self.opt.zero_grad(set_to_none=True)
            for micro in range(accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    self.state.epoch += 1
                    self.state.batches_in_epoch = 0
                    if sampler is not None:
                        sampler.set_epoch(self.state.epoch)
                    data_iter = iter(loader)
                    batch = next(data_iter)
                self.state.batches_in_epoch += 1
                batch = self._to_device(batch)
                logs = self.net(batch, stage=self.stage)
                (logs["loss"] / accum).backward()
            gnorm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], tc.grad_clip
            )
            # Gradients are all-reduced, so gnorm is the same on every rank and
            # every rank takes the same branch. A NaN/inf step is skipped, not
            # applied; the step still counts (its batches were consumed).
            if torch.isfinite(gnorm):
                self.opt.step()
            else:
                self.skipped_steps += 1
                self.log_fn(f"[stage {self.stage}] step {self.state.step + 1}: non-finite "
                            f"gradient norm, update skipped ({self.skipped_steps} so far)")
            self.state.step += 1
            last_logs = {k: float(v.detach()) for k, v in logs.items()}

            if self.state.step % tc.log_every == 0:
                msg = " ".join(f"{k}={v:.4f}" for k, v in last_logs.items())
                self.log_fn(
                    f"[stage {self.stage}] step {self.state.step}/{max_steps} "
                    f"lr={lr:.2e} {msg} ({time.time() - t0:.1f}s)"
                )
            if self.ckpt_dir and self.rank == 0 and self.state.step % tc.ckpt_every == 0:
                self.save(self.ckpt_dir / f"step_{self.state.step:07d}.pt")
        if self.is_ddp:
            dist.barrier()
        return last_logs

    @staticmethod
    def _data_position(sampler, loader, micro_batches: int) -> tuple[int, int]:
        """(epoch, batches consumed in it) after `micro_batches` loader batches."""
        bs = loader.batch_size
        epoch = 0
        while True:
            sampler.set_epoch(epoch)
            n = len(sampler)
            per_epoch = n // bs if loader.drop_last else -(-n // bs)
            if per_epoch <= 0:
                raise RuntimeError(f"epoch {epoch} yields no batches (n={n}, batch={bs})")
            if micro_batches < per_epoch:
                return epoch, micro_batches
            micro_batches -= per_epoch
            epoch += 1

    def save(self, path: Path) -> None:
        """Atomic: a wall-time kill mid-write must never leave a truncated
        step_*.pt for the next chained job to resume from."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        torch.save({
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "step": self.state.step,
            "epoch": self.state.epoch,
            "batches_in_epoch": self.state.batches_in_epoch,
            "config": self.cfg.to_dict(),
            "run": dict(self.run_meta),
        }, tmp)
        os.replace(tmp, path)
        self._prune(path.parent)

    def _prune(self, ckpt_dir: Path) -> None:
        """Keep every keep_every-th checkpoint plus the newest two."""
        keep_every = self.cfg.train.keep_every
        if keep_every <= 0:
            return
        ckpts = sorted(ckpt_dir.glob("step_*.pt"))
        for c in ckpts[:-2]:
            step = int(c.stem.split("_")[1])
            if step % keep_every:
                c.unlink(missing_ok=True)

    # Fields that may legitimately differ between a run and its resumption.
    RESUME_MUTABLE = {("train", "ckpt_every"), ("train", "log_every")}

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        saved = ckpt.get("config")
        if saved is not None:
            now = self.cfg.to_dict()
            diff = [f"{sec}.{k}: checkpoint={v!r} now={now.get(sec, {}).get(k)!r}"
                    for sec, fields in saved.items() for k, v in fields.items()
                    if (sec, k) not in self.RESUME_MUTABLE
                    and _norm(now.get(sec, {}).get(k)) != _norm(v)]
            if diff:
                raise SystemExit(
                    f"Refusing to resume {path}: config differs from the run that wrote it "
                    f"(pass the same --config/--set as the original job):\n  "
                    + "\n  ".join(diff))
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["opt"])
        self.state.step = ckpt["step"]
        self._resumed_meta = ckpt.get("run") or None
        # The data position is recomputed from the step in fit() (per rank);
        # the saved values are informational (rank 0's position).
        self.state.epoch = int(ckpt.get("epoch", 0))
        self.state.batches_in_epoch = int(ckpt.get("batches_in_epoch", 0))


def _norm(v):
    """Compare configs across YAML/JSON round trips (tuples vs lists)."""
    return list(v) if isinstance(v, (list, tuple)) else v
