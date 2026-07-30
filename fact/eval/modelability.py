"""Modelability probe #1: LM-probe bits-per-frame.

"How easy is this token stream for a downstream generator?" - train a small
fixed-budget causal LM on a tokenizer's stream and report held-out NLL,
normalized to bits per frame AND bits per second (so different frame rates
compare fairly). This is the first controlled cross-tokenizer number of its
kind; the protocol is tokenizer-agnostic - anything that yields per-frame
integer indices (one or more streams) can be probed.

For FaCT the probe models the factored stream jointly:
    p(content_t, prosody_t | tokens_<t) = p(c_t | ...) * p(p_t | ..., c_t)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..modules.transformer import CausalTransformer


@dataclass
class ModelabilityResult:
    bits_per_frame: float
    bits_per_second: float
    frames_evaluated: int
    breakdown: dict[str, float]  # per-stream bits/frame


class TokenLMProbe(nn.Module):
    """Fixed small causal LM over one or more parallel index streams.

    Streams are embedded, summed, contextualized causally; each stream gets
    its own prediction head. Stream k at frame t is predicted from all
    streams at frames < t, plus (chain rule) earlier streams at frame t via
    conditioning heads.
    """

    def __init__(self, vocab_sizes: list[int], dim: int = 512,
                 n_layers: int = 8, n_heads: int = 8):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.embeds = nn.ModuleList(nn.Embedding(v, dim) for v in vocab_sizes)
        self.start = nn.Parameter(torch.zeros(1, 1, dim))
        self.transformer = CausalTransformer(dim, n_layers, n_heads)
        # Head k conditions on the hidden state plus embeddings of streams < k
        # at the same frame (chain-rule factorization).
        self.heads = nn.ModuleList(
            nn.Linear(dim, v) for v in vocab_sizes
        )
        self.cond_proj = nn.ModuleList(
            nn.Linear(dim, dim) if k > 0 else nn.Identity()
            for k in range(len(vocab_sizes))
        )

    def stream_logits(self, streams: list[torch.Tensor]) -> list[torch.Tensor]:
        """streams: list of (B, T) long tensors -> list of (B, T, V_k) logits."""
        b, t = streams[0].shape
        x = sum(emb(s) for emb, s in zip(self.embeds, streams))
        # Shift right: frame t predicted from frames < t.
        x = torch.cat([self.start.expand(b, 1, -1), x[:, :-1]], dim=1)
        h = self.transformer(x)
        logits = []
        cond = h
        for k, head in enumerate(self.heads):
            if k > 0:
                # Chain rule within the frame: add embedding of stream k-1 at t.
                cond = cond + self.cond_proj[k](self.embeds[k - 1](streams[k - 1]))
            logits.append(head(cond))
        return logits

    def loss(self, streams: list[torch.Tensor]) -> torch.Tensor:
        logits = self.stream_logits(streams)
        losses = [
            F.cross_entropy(lg.transpose(1, 2), s) for lg, s in zip(logits, streams)
        ]
        return sum(losses)


@torch.no_grad()
def evaluate_bits(probe: TokenLMProbe, batches: Iterable[list[torch.Tensor]],
                  frame_rate_hz: float) -> ModelabilityResult:
    probe.eval()
    total_nll = torch.zeros(len(probe.vocab_sizes))
    n_frames = 0
    for streams in batches:
        logits = probe.stream_logits(streams)
        for k, (lg, s) in enumerate(zip(logits, streams)):
            total_nll[k] += F.cross_entropy(
                lg.transpose(1, 2), s, reduction="sum"
            ).item()
        n_frames += streams[0].numel()
    ln2 = torch.log(torch.tensor(2.0)).item()
    per_stream = {f"stream_{k}": float(total_nll[k]) / n_frames / ln2
                  for k in range(len(probe.vocab_sizes))}
    bpf = sum(per_stream.values())
    return ModelabilityResult(
        bits_per_frame=bpf,
        bits_per_second=bpf * frame_rate_hz,
        frames_evaluated=n_frames,
        breakdown=per_stream,
    )


def train_probe(probe: TokenLMProbe, batches: Iterable[list[torch.Tensor]],
                steps: int, lr: float = 3e-4,
                device: str = "cpu") -> TokenLMProbe:
    """Fixed-budget probe training: same steps/data for every tokenizer."""
    probe = probe.to(device).train()
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    it = iter(batches)
    for _ in range(steps):
        try:
            streams = next(it)
        except StopIteration:
            it = iter(batches)
            streams = next(it)
        streams = [s.to(device) for s in streams]
        loss = probe.loss(streams)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return probe
