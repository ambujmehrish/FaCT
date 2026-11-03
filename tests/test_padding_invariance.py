"""Padded batches: with lengths, every item is encoded and decoded as if it
were alone (audit 2026-11-03, C: non-causal arms used to attend to padding)."""

import dataclasses

import torch

from fact.config import tiny_config
from fact.model import FaCT


def _pair(causal: bool):
    cfg = tiny_config()
    cfg.encoder = dataclasses.replace(cfg.encoder, causal=causal)
    cfg.decoder = dataclasses.replace(cfg.decoder, causal=causal)
    torch.manual_seed(0)
    m = FaCT(cfg).eval()
    s = cfg.encoder.frame_stack
    short = torch.randn(1, 6 * s, cfg.audio.n_mels) - 4.0
    long_ = torch.randn(1, 15 * s, cfg.audio.n_mels) - 4.0
    batch = torch.full((2, 15 * s, cfg.audio.n_mels), cfg.audio.mel_mean)  # padding value of collate
    batch[0, : short.shape[1]] = short[0]
    batch[1] = long_[0]
    return m, short, batch, torch.tensor([short.shape[1], long_.shape[1]])


@torch.no_grad()
def test_noncausal_encoder_ignores_padding_with_lengths():
    m, short, batch, lengths = _pair(causal=False)
    t = short.shape[1] // m.cfg.encoder.frame_stack
    alone = m.encoder(m._norm_mel(short))
    padded = m.encoder(m._norm_mel(batch), lengths=lengths)[:1, :t]
    assert torch.allclose(alone, padded, atol=1e-5)
    # Without lengths the padding leaks into every frame (the old behaviour).
    leaky = m.encoder(m._norm_mel(batch))[:1, :t]
    assert not torch.allclose(alone, leaky, atol=1e-3)


@torch.no_grad()
def test_noncausal_decoder_ignores_padding_with_key_mask():
    m, short, batch, lengths = _pair(causal=False)
    n = short.shape[1]
    tok = n // m.cfg.encoder.frame_stack
    cond = torch.randn(2, 15, m.decoder.cond_proj.in_features)
    spk = torch.randn(2, m.decoder.spk_proj.in_features)
    x = torch.randn(2, batch.shape[1], m.cfg.audio.n_mels)
    t = torch.full((2,), 0.5)
    alone = m.decoder(x[:1, :n], t[:1], cond[:1, :tok], spk[:1])
    key = torch.arange(batch.shape[1])[None] < lengths[:, None]
    padded = m.decoder(x, t, cond, spk, key_mask=key)[:1, :n]
    assert torch.allclose(alone, padded, atol=1e-5)


@torch.no_grad()
def test_causal_encoder_unchanged_by_lengths():
    """Strictly causal: valid frames never saw the padded tail, so passing
    lengths changes nothing for them (training of causal arms is unchanged)."""
    m, short, batch, lengths = _pair(causal=True)
    t = short.shape[1] // m.cfg.encoder.frame_stack
    a = m.encoder(m._norm_mel(batch))[:1, :t]
    b = m.encoder(m._norm_mel(batch), lengths=lengths)[:1, :t]
    assert torch.equal(a, b)
