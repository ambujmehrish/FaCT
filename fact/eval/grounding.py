"""Content-grounding and codebook metrics on padded batches.

Shared by the lambda_ctc sweep and the gate tripwires. Batches are padded
to their longest item, and the mel pad value is mel_mean, which is not
silence. So every metric here only looks at each item's VALID tokens:
padding frames would otherwise add spurious CTC emissions to the CER and
phantom codes to the codebook statistics.
"""

from __future__ import annotations

import math

import torch

from .wer import word_edit_distance


def token_lengths(batch, t_tok: int, frame_stack: int) -> torch.Tensor:
    """Per-item valid token count; same rule as FaCT.training_step."""
    if batch.mel_lengths is None:
        return torch.full((batch.mel.shape[0],), t_tok, dtype=torch.long)
    return (batch.mel_lengths.cpu() // frame_stack).clamp(min=1, max=t_tok)


def valid_indices(indices: torch.Tensor, lengths: torch.Tensor,
                  rate_divisor: int = 1) -> torch.Tensor:
    """Flatten (B, T') indices to the valid positions only.

    rate_divisor > 1 is the slow prosody stream: T' = ceil(T / r), so an
    item with n valid tokens has ceil(n / r) valid prosody positions.
    """
    idx = indices.cpu()
    keep = [(int(n) + rate_divisor - 1) // rate_divisor for n in lengths]
    return torch.cat([idx[i, :k] for i, k in enumerate(keep)])


def codebook_stats(indices: torch.Tensor, codebook_size: int) -> dict:
    """Utilization (fraction of codes seen) and perplexity (exp entropy).

    Utilization is bounded by n_tokens / codebook_size, so evaluate on
    enough tokens (n_tokens >> codebook_size) for it to mean anything;
    `tokens` is reported for that check.
    """
    counts = torch.bincount(indices.flatten(), minlength=codebook_size).float()
    probs = counts / counts.sum().clamp_min(1)
    nz = probs[probs > 0]
    return {
        "codebook_size": codebook_size,
        "tokens": int(counts.sum()),
        "codes_used": int((counts > 0).sum()),
        "utilization": float((counts > 0).float().mean()),
        "perplexity": float(math.exp(float(-(nz * nz.log()).sum()))),
    }


@torch.no_grad()
def greedy_byte_cer(model, batches: list, device: str) -> float:
    """Greedy-CTC byte error rate of the content stream (valid frames only)."""
    total_dist = total_ref = 0
    s = model.cfg.encoder.frame_stack
    for batch in batches:
        out = model.encode_mel(batch.mel.to(device))
        emb = out.primary_emb()
        lengths = token_lengths(batch, emb.shape[1], s)
        hyps = model.ctc_head.greedy_decode(emb, token_lengths=lengths)
        for i, hyp in enumerate(hyps):
            n = int(batch.text_lengths[i])
            if n == 0:
                continue
            total_dist += word_edit_distance(batch.text[i, :n].tolist(), hyp)
            total_ref += n
    return total_dist / max(1, total_ref)
