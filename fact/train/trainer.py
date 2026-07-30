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


class Trainer:
    def __init__(self, model: FaCT, cfg: FaCTConfig, stage: str = "a",
                 device: str = "cpu", ckpt_dir: Optional[str] = None,
                 log_fn: Callable[[str], None] = print):
        self.is_ddp, self.rank, self.world_size = ddp_env()
        self.model = model.to(device)
        self.cfg = cfg
        self.stage = stage
        self.device = device
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.log_fn = log_fn if self.rank == 0 else (lambda s: None)
        self.state = TrainState()

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
            ref_mel=mv(batch.ref_mel),
        )

    def fit(self, loader: Iterable[Batch], max_steps: Optional[int] = None,
            sampler=None) -> dict[str, float]:
        tc = self.cfg.train
        max_steps = max_steps or tc.max_steps
        accum = max(1, tc.grad_accum)
        self.net.train()
        last_logs: dict[str, float] = {}
        t0 = time.time()
        epoch = 0

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
                    epoch += 1
                    if sampler is not None:
                        sampler.set_epoch(epoch)
                    data_iter = iter(loader)
                    batch = next(data_iter)
                batch = self._to_device(batch)
                logs = self.net(batch, stage=self.stage)
                (logs["loss"] / accum).backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], tc.grad_clip
            )
            self.opt.step()
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

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "step": self.state.step,
            "config": self.cfg.to_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["opt"])
        self.state.step = ckpt["step"]
