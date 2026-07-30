"""Causality is a headline claim - verify it mechanically.

Perturb inputs at time >= T and assert outputs at time < T are unchanged
(encoder: strictly causal at token granularity; decoder: block-causal).
"""

import torch

from fact.config import tiny_config
from fact.model import FaCT
from fact.modules.transformer import block_causal_mask, causal_mask


def test_causal_mask_shapes():
    m = causal_mask(5, torch.device("cpu"))
    assert m[0, 1] == False and m[1, 0] == True  # noqa: E712


def test_block_causal_mask():
    m = block_causal_mask(8, 4, torch.device("cpu"))
    assert m[0, 3] == True   # same block: full attention  # noqa: E712
    assert m[0, 4] == False  # future block: masked        # noqa: E712
    assert m[7, 0] == True   # past block visible          # noqa: E712


def test_encoder_is_causal():
    cfg = tiny_config()
    model = FaCT(cfg).eval()
    torch.manual_seed(0)
    t_mel = 32  # 8 tokens at frame_stack 4
    mel = torch.randn(1, t_mel, cfg.audio.n_mels)
    mel2 = mel.clone()
    mel2[:, 16:] += 10.0  # perturb tokens 4..7 only
    with torch.no_grad():
        h1 = model.encoder(mel)
        h2 = model.encoder(mel2)
    assert torch.allclose(h1[:, :4], h2[:, :4], atol=1e-4)
    assert not torch.allclose(h1[:, 4:], h2[:, 4:], atol=1e-2)


def test_decoder_is_block_causal():
    cfg = tiny_config()
    model = FaCT(cfg).eval()
    torch.manual_seed(0)
    b, t_tok = 1, 8
    t_mel = t_tok * cfg.encoder.frame_stack  # 32 mel frames, block_size 4
    x_t = torch.randn(b, t_mel, cfg.audio.n_mels)
    cond = torch.randn(b, t_tok, cfg.encoder.dim)
    spk = torch.randn(b, cfg.speaker.dim)
    t = torch.tensor([0.5])

    cond2 = cond.clone()
    cond2[:, 4:] += 10.0   # perturb condition from token 4 (mel frame 16)
    x2 = x_t.clone()
    x2[:, 16:] += 10.0
    with torch.no_grad():
        v1 = model.decoder(x_t, t, cond, spk)
        v2 = model.decoder(x2, t, cond2, spk)
    assert torch.allclose(v1[:, :16], v2[:, :16], atol=1e-4)


def test_tokens_independent_of_future():
    """tokenize() at token i must not change when future audio changes."""
    cfg = tiny_config()
    model = FaCT(cfg).eval()
    torch.manual_seed(1)
    mel = torch.randn(1, 40, cfg.audio.n_mels)
    mel2 = mel.clone()
    mel2[:, 20:] = torch.randn_like(mel2[:, 20:]) * 5
    t1 = model.tokenize(mel=mel)
    t2 = model.tokenize(mel=mel2)
    assert torch.equal(t1.content_indices[:, :5], t2.content_indices[:, :5])
    assert torch.equal(t1.prosody_indices[:, :5], t2.prosody_indices[:, :5])
