"""Causal stage-D / vocoder generator (VocoderConfig.causal): output samples
depend on input frames up to 2 frames (the iSTFT overlap) ahead, never more;
the default (centered) generator keeps its parameters and behaviour."""

import dataclasses

import pytest
import torch

from fact.vocoder.model import Generator, VocoderConfig, tiny_vocoder_config


def _gen(causal, lookahead=0):
    torch.manual_seed(0)
    cfg = dataclasses.replace(tiny_vocoder_config(), causal=causal, lookahead=lookahead)
    return Generator(cfg, in_dim=16).eval(), cfg


def test_causal_generator_lookahead_is_the_istft_overlap():
    g, cfg = _gen(True)
    hop = cfg.audio.hop_length
    x = torch.randn(1, 40, 16)
    base = g(x)
    for k in (10, 25):
        y = x.clone()
        y[:, k:] += torch.randn_like(y[:, k:])
        out = g(y)
        cut = (k - 2) * hop  # samples before this cannot see frame k
        assert torch.allclose(out[:, :cut], base[:, :cut], atol=1e-5)
        assert not torch.allclose(out[:, cut:], base[:, cut:], atol=1e-5)


def test_centered_generator_looks_ahead_and_old_checkpoints_load():
    g, cfg = _gen(False)
    hop = cfg.audio.hop_length
    x = torch.randn(1, 40, 16)
    y = x.clone()
    y[:, 25:] += torch.randn_like(y[:, 25:])
    # Samples 1-5 frames BEFORE the change already move (centered convs).
    assert (g(y) - g(x))[:, 20 * hop: 24 * hop].abs().max() > 1e-4
    gc, _ = _gen(True)
    gc.load_state_dict(g.state_dict())  # same parameters either way
    assert VocoderConfig.from_dict({k: v for k, v in dataclasses.asdict(cfg).items()
                                    if k != "causal"}).causal is False


@pytest.mark.parametrize("la", [2, 8])
def test_lookahead_generator_sees_exactly_la_frames_ahead(la):
    g, cfg = _gen(True, la)  # tiny: 2 blocks -> up to 18 frames; 8 spans embed + block 0
    hop = cfg.audio.hop_length
    x = torch.randn(1, 40, 16)
    base = g(x)
    k = 25
    y = x.clone()
    y[:, k:] += torch.randn_like(y[:, k:])
    out = g(y)
    cut = (k - la - 2) * hop  # samples before this cannot see frame k
    assert torch.allclose(out[:, :cut], base[:, :cut], atol=1e-5)
    assert not torch.allclose(out[:, cut: cut + 2 * hop], base[:, cut: cut + 2 * hop], atol=1e-6)


def test_lookahead_config_checks():
    with pytest.raises(ValueError):
        _gen(False, 2)  # centered convs + lookahead
    with pytest.raises(ValueError):
        _gen(True, 100)  # more than the convolutions can provide
