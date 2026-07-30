"""Minimal single-node trainer for stages A and B.

Plain DDP-ready loop (wrap the model in torch.nn.parallel.DistributedDataParallel
and use a DistributedSampler for multi-GPU; the loop itself is agnostic).
Stage A trains everything with the full loss; stage B freezes the tokenizer
and shortcut-distills the decoder (SiTok recipe).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
from torch.utils.data import DataLoader

from ..config import FaCTConfig
from ..model import Batch, FaCT


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
        self.model = model.to(device)
        self.cfg = cfg
        self.stage = stage
        self.device = device
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.log_fn = log_fn
        self.state = TrainState()

        if stage == "b":
            self.model.freeze_tokenizer()
        params = [p for p in self.model.parameters() if p.requires_grad]
        tc = cfg.train
        self.opt = torch.optim.AdamW(
            params, lr=tc.lr, betas=tc.betas, weight_decay=tc.weight_decay
        )

    def _to_device(self, batch: Batch) -> Batch:
        def mv(x):
            return x.to(self.device) if isinstance(x, torch.Tensor) else x
        return Batch(
            mel=mv(batch.mel), text=mv(batch.text), text_lengths=mv(batch.text_lengths),
            f0_bins=mv(batch.f0_bins), energy=mv(batch.energy),
            ref_mel=mv(batch.ref_mel),
        )

    def fit(self, loader: Iterable[Batch], max_steps: Optional[int] = None) -> dict[str, float]:
        tc = self.cfg.train
        max_steps = max_steps or tc.max_steps
        self.model.train()
        last_logs: dict[str, float] = {}
        t0 = time.time()

        data_iter = iter(loader)
        while self.state.step < max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            batch = self._to_device(batch)

            lr = cosine_lr(self.state.step, tc.warmup_steps, max_steps, tc.lr)
            for g in self.opt.param_groups:
                g["lr"] = lr

            logs = self.model.training_step(batch, stage=self.stage)
            self.opt.zero_grad(set_to_none=True)
            logs["loss"].backward()
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
            if self.ckpt_dir and self.state.step % tc.ckpt_every == 0:
                self.save(self.ckpt_dir / f"step_{self.state.step:07d}.pt")
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
