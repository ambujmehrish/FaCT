"""Chunked-lookahead encoder (encoder.chunk_tokens): token i depends on mel
frames up to the end of its chunk and on nothing later; chunk_tokens=1 is
the original strictly causal encoder."""

import pytest
import torch

from fact.config import tiny_config
from fact.modules.encoder import CausalEncoder


def _enc(chunk):
    cfg = tiny_config()
    cfg.encoder.chunk_tokens = chunk
    torch.manual_seed(0)
    return CausalEncoder(cfg.audio, cfg.encoder).eval(), cfg


@pytest.mark.parametrize("chunk", [1, 2, 3])
def test_dependence_ends_at_chunk_boundary(chunk):
    enc, cfg = _enc(chunk)
    s = cfg.encoder.frame_stack
    n_tok = 4 * chunk
    mel = torch.randn(1, n_tok * s, cfg.audio.n_mels)
    base = enc(mel)
    for i in range(n_tok):
        end = (i // chunk + 1) * chunk * s  # first frame token i must NOT see
        if end < mel.shape[1]:
            later = mel.clone()
            later[:, end:] += torch.randn_like(later[:, end:])
            assert torch.allclose(enc(later)[0, i], base[0, i], atol=1e-5), (chunk, i)
        if end - 1 >= (i + 1) * s:  # a future frame inside the chunk is seen
            inside = mel.clone()
            inside[:, end - 1] += 5.0
            assert not torch.allclose(enc(inside)[0, i], base[0, i], atol=1e-5), (chunk, i)


def test_chunk_one_is_the_original_causal_encoder():
    enc1, cfg = _enc(1)
    assert enc1.pre.block_size is None and enc1.post.block_size is None
    s = cfg.encoder.frame_stack
    mel = torch.randn(1, 6 * s, cfg.audio.n_mels)
    a = enc1(mel)
    later = mel.clone()
    later[:, 3 * s:] += 1.0
    assert torch.allclose(enc1(later)[0, :3], a[0, :3], atol=1e-5)


def test_chunk_weights_are_interchangeable_and_config_checked():
    enc1, cfg = _enc(1)
    enc2, _ = _enc(2)
    enc2.load_state_dict(enc1.state_dict())  # fine-tuning a causal checkpoint
    cfg.encoder.chunk_tokens = 0
    with pytest.raises(ValueError):
        CausalEncoder(cfg.audio, cfg.encoder)
    cfg.encoder.chunk_tokens, cfg.encoder.causal = 2, False
    with pytest.raises(ValueError):
        CausalEncoder(cfg.audio, cfg.encoder)
