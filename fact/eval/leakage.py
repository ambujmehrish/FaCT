"""Modelability probe #3: the disentanglement leakage matrix (DSA-style).

Train independent post-hoc probes on *frozen* tokenizer outputs and report
accuracies:

                 -> F0 class   -> text (frame)   -> speaker id
    content        LOW (leak)     high (good)      low (good)
    prosody        high (good)    LOW (leak)       low (good)

Unlike the GRL probes used during training, these are trained fresh with
detached features - they measure what is *recoverable*, not what the
adversary happened to learn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class LeakageMatrix:
    content_to_f0_acc: float
    prosody_to_f0_acc: float
    content_to_text_acc: float
    prosody_to_text_acc: float
    content_to_speaker_acc: float
    prosody_to_speaker_acc: float

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


class FrameProbe(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_probe(probe: FrameProbe,
                 make_batches: Callable[[], Iterable[tuple[torch.Tensor, torch.Tensor]]],
                 steps: int, lr: float, pooled: bool) -> None:
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    it = iter(make_batches())
    for _ in range(steps):
        try:
            feats, target = next(it)
        except StopIteration:
            it = iter(make_batches())
            feats, target = next(it)
        feats = feats.detach()
        if pooled:
            logits = probe(feats.mean(dim=1))
            loss = F.cross_entropy(logits, target)
        else:
            logits = probe(feats)
            loss = F.cross_entropy(logits.transpose(1, 2), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


@torch.no_grad()
def _probe_acc(probe: FrameProbe,
               batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
               pooled: bool) -> float:
    correct = total = 0
    for feats, target in batches:
        if pooled:
            pred = probe(feats.mean(dim=1)).argmax(-1)
        else:
            pred = probe(feats).argmax(-1)
        correct += (pred == target).sum().item()
        total += target.numel()
    return correct / max(1, total)


def probe_accuracy(
    make_batches: Callable[[], Iterable[tuple[torch.Tensor, torch.Tensor]]],
    in_dim: int, n_classes: int,
    steps: int = 500, lr: float = 1e-3, pooled: bool = False,
) -> float:
    """Train a fresh probe on (features, targets) batches; return eval accuracy.

    `make_batches` is called twice: once for training, once for evaluation
    (pass a factory over train/held-out splits as appropriate).
    """
    probe = FrameProbe(in_dim, n_classes)
    _train_probe(probe, make_batches, steps, lr, pooled)
    return _probe_acc(probe, make_batches(), pooled)
