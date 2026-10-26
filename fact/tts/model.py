"""Decoder-only TTS LM over multi-stream codec tokens (identical for every codec).

Sequence: [text bytes] [EOT] [audio frames]. Audio frames carry S parallel
streams (FaCT: content + prosody; Mimi-2: codebooks 0 and 1); stream s is
delayed by delays[s] frames (MusicGen delay pattern, default 0, 1, ...), so
stream s at frame t is predicted after seeing streams < s at frame t. The
input embedding of an audio position is the sum of its stream embeddings;
each stream has its own output head over [codes..., EOS].

Per stream s with V_s codes the embedding ids are: codes 0..V_s-1,
EOS = V_s, PAD = V_s + 1 (delay padding / after EOS), BOS = V_s + 2.
Targets equal to PAD carry no loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..modules.transformer import RMSNorm, SwiGLU

TEXT_VOCAB = 258  # 0 pad, 1..256 bytes + 1, 257 EOT
TEXT_PAD, EOT = 0, 257


@dataclass
class TTSConfig:
    vocab_sizes: tuple = (8192, 625)
    delays: Optional[tuple] = None  # default (0, 1, ..., S-1)
    dim: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    ffn_mult: float = 4.0
    dropout: float = 0.0
    max_text: int = 400
    max_frames: int = 400
    rope_base: float = 10_000.0

    def __post_init__(self):
        self.vocab_sizes = tuple(int(v) for v in self.vocab_sizes)
        if self.delays is None:
            self.delays = tuple(range(len(self.vocab_sizes)))
        self.delays = tuple(int(d) for d in self.delays)
        if len(self.delays) != len(self.vocab_sizes) or min(self.delays) != 0:
            raise ValueError("one delay per stream, the smallest 0")

    @property
    def n_streams(self) -> int:
        return len(self.vocab_sizes)

    @property
    def max_delay(self) -> int:
        return max(self.delays)

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def eos(v: int) -> int:
    return v


def pad_id(v: int) -> int:
    return v + 1


def bos(v: int) -> int:
    return v + 2


def delayed_targets(codes: torch.Tensor, cfg: TTSConfig) -> torch.Tensor:
    """codes (S, T) -> targets (S, T + D + 1) with delay padding and EOS."""
    s_n, t = codes.shape
    d = cfg.max_delay
    y = torch.empty(s_n, t + d + 1, dtype=torch.long)
    for s, (v, ds) in enumerate(zip(cfg.vocab_sizes, cfg.delays)):
        y[s].fill_(pad_id(v))
        y[s, ds: ds + t] = codes[s].long()
        y[s, ds + t] = eos(v)
    return y


def undelay(y: torch.Tensor, cfg: TTSConfig, t: int) -> torch.Tensor:
    """Targets (S, >= t + D) -> codes (S, t): inverse of the delay pattern."""
    return torch.stack([y[s, ds: ds + t] for s, ds in enumerate(cfg.delays)])


# ------------------------------------------------------------------- model

def _rope(x: torch.Tensor, pos: torch.Tensor, base: float) -> torch.Tensor:
    """x (B, H, T, Dh), pos (B, T) integer positions."""
    dh = x.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, dh, 2, device=x.device, dtype=torch.float32) / dh))
    ang = pos.float()[..., None] * inv  # (B, T, Dh/2)
    cos, sin = ang.cos()[:, None], ang.sin()[:, None]
    x1, x2 = x.float()[..., 0::2], x.float()[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
    return out.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: TTSConfig):
        super().__init__()
        self.h = cfg.n_heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.base = cfg.rope_base
        self.dropout = cfg.dropout

    def forward(self, x, pos, mask=None, cache=None):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k = _rope(q, pos, self.base), _rope(k, pos, self.base)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class Block(nn.Module):
    def __init__(self, cfg: TTSConfig):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.dim), RMSNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.ffn = SwiGLU(cfg.dim, cfg.ffn_mult)

    def forward(self, x, pos, mask=None, cache=None):
        x = x + self.attn(self.n1(x), pos, mask, cache)
        return x + self.ffn(self.n2(x))


class TTSLM(nn.Module):
    def __init__(self, cfg: TTSConfig):
        super().__init__()
        self.cfg = cfg
        self.text_emb = nn.Embedding(TEXT_VOCAB, cfg.dim)
        self.code_emb = nn.ModuleList(nn.Embedding(v + 3, cfg.dim) for v in cfg.vocab_sizes)
        self.type_emb = nn.Embedding(2, cfg.dim)  # 0 text, 1 audio
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim)
        self.heads = nn.ModuleList(nn.Linear(cfg.dim, v + 1, bias=False) for v in cfg.vocab_sizes)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def embed(self, text: torch.Tensor, audio_in: torch.Tensor) -> torch.Tensor:
        """text (B, L) ids, audio_in (B, S, A) ids -> (B, L + A, dim)."""
        a = sum(emb(audio_in[:, s]) for s, emb in enumerate(self.code_emb))
        t = self.text_emb(text) + self.type_emb.weight[0]
        return torch.cat([t, a + self.type_emb.weight[1]], dim=1)

    def forward(self, text, audio_in, valid) -> list[torch.Tensor]:
        """Logits per stream over the audio positions: S x (B, A, V_s + 1).
        valid (B, L + A) bool marks real (non-padding) positions."""
        x = self.embed(text, audio_in)
        pos = (valid.long().cumsum(1) - 1).clamp(min=0)
        n = x.shape[1]
        causal = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
        mask = causal[None] & valid[:, None, :]
        mask = mask | torch.eye(n, dtype=torch.bool, device=x.device)[None]  # no empty rows
        mask = mask[:, None]
        for blk in self.blocks:
            x = blk(x, pos, mask)
        h = self.norm(x[:, text.shape[1]:])
        return [head(h) for head in self.heads]

    def loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        logits = self(batch["text"], batch["audio_in"], batch["valid"])
        total, logs = 0.0, {}
        for s, (lg, v) in enumerate(zip(logits, self.cfg.vocab_sizes)):
            y = batch["targets"][:, s]
            keep = y != pad_id(v)
            ce = F.cross_entropy(lg[keep].float(), y[keep])
            total = total + ce
            logs[f"ce{s}"] = float(ce.detach())
        return total / len(logits), logs

    # ------------------------------------------------------------ inference

    @torch.no_grad()
    def generate(self, text: torch.Tensor, prompt_codes: torch.Tensor, max_new_frames: int = 400,
                 temperature: float = 1.0, top_k: int = 50, top_p: float = 1.0,
                 min_new_frames: int = 1, generator: Optional[torch.Generator] = None,
                 fixed_stream0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """text (L,) ids = prompt transcript + target transcript; prompt_codes
        (S, Tp). Returns the NEW frames (S, T_new). Stream 0 decides the
        length; the other streams end exactly at the same frame.
        fixed_stream0 (T_new,): force stream 0's new frames (and so the length)
        and sample only the other streams - e.g. resample FaCT's prosody for
        fixed content."""
        cfg = self.cfg
        dev = next(self.parameters()).device
        s_n, tp = prompt_codes.shape
        d = cfg.max_delay
        vs, ds = cfg.vocab_sizes, cfg.delays
        txt = torch.cat([text.long(), torch.tensor([EOT])]).to(dev)[None]
        caches = [dict() for _ in self.blocks]

        def run(x, start):
            pos = torch.arange(start, start + x.shape[1], device=dev)[None]
            n = x.shape[1]
            mask = None
            if n > 1:  # prefill: causal within the chunk (the cache is empty)
                mask = torch.ones(n, n, dtype=torch.bool, device=dev).tril()[None, None]
            for blk, c in zip(self.blocks, caches):
                x = blk(x, pos, mask, c)
            return self.norm(x[:, -1])

        # y[s][j]: target sequence; known from the prompt where j - d_s < Tp.
        y = [[None] * 0 for _ in range(s_n)]

        def known(s, j):
            t = j - ds[s]
            if t < 0:
                return pad_id(vs[s])
            if t < tp:
                return int(prompt_codes[s, t])
            return None

        x = self.embed(txt, torch.tensor([[bos(v)] for v in vs], device=dev)[None])
        h = run(x, 0)
        pos = x.shape[1]
        end0 = None  # frame index where stream 0 emitted EOS
        fixed = None
        if fixed_stream0 is not None:
            fixed = fixed_stream0.long().cpu()
            if fixed.numel() < 1 or int(fixed.max()) >= vs[0] or int(fixed.min()) < 0:
                raise ValueError("fixed_stream0 must hold >= 1 valid stream-0 codes")
            end0 = tp + fixed.numel()
        j = 0
        while True:
            logits = [head(h)[0].float() for head in self.heads]
            col = []
            for s in range(s_n):
                k = known(s, j)
                t = j - ds[s]
                if k is None and s == 0 and fixed is not None and t < end0:
                    k = int(fixed[t - tp])
                if k is not None:
                    col.append(k)
                    continue
                if end0 is not None and t >= end0:
                    col.append(eos(vs[s]) if t == end0 else pad_id(vs[s]))
                    continue
                lg = logits[s].clone()
                if s > 0 or t - tp < min_new_frames:
                    lg[eos(vs[s])] = -float("inf")  # only stream 0 may end, not too early
                col.append(_sample(lg, temperature, top_k, top_p, generator))
                if s == 0 and col[-1] == eos(vs[0]):
                    end0 = t
            for s in range(s_n):
                y[s].append(col[s])
            if end0 is None and j - ds[0] - tp + 1 >= max_new_frames:
                end0 = j - ds[0] + 1  # length cap: stream 0 ends after this frame
            if end0 is not None and j >= end0 + d:
                break
            inp = torch.tensor([[[c]] for c in col], device=dev).transpose(0, 1)  # (1, S, 1)
            h = run(self.embed(txt[:, :0], inp), pos)
            pos += 1
            j += 1
        ty = torch.tensor(y)
        return undelay(ty, cfg, end0)[:, tp:]


def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float,
            generator: Optional[torch.Generator]) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    lg = logits / temperature
    if top_k and top_k < lg.shape[0]:
        kth = torch.topk(lg, top_k).values[-1]
        lg = lg.masked_fill(lg < kth, -float("inf"))
    p = torch.softmax(lg, dim=-1)
    if top_p < 1.0:
        sp, si = torch.sort(p, descending=True)
        cut = sp.cumsum(0) - sp > top_p
        sp[cut] = 0
        p = torch.zeros_like(p).scatter_(0, si, sp)
        p = p / p.sum()
    return int(torch.multinomial(p.cpu(), 1, generator=generator))


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
