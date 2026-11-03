"""Algorithmic latency: the formula (fact/eval/latency.py) matches a
perturbation measurement of the real encoder + generator code path."""

import dataclasses

import pytest
import torch

from fact.config import AudioConfig, tiny_config
from fact.eval.latency import algorithmic_latency_ms, measure_ms
from fact.model import FaCT
from fact.modules.mel import LogMelSpectrogram
from fact.train.stage_d_wave_decoder import input_dim
from fact.vocoder.model import Generator, VocoderConfig


def _setup(la):
    cfg = tiny_config()
    cfg.audio = AudioConfig()  # 24 kHz, hop 480, win 960, n_fft 1024
    torch.manual_seed(0)
    fact = FaCT(cfg).eval()
    vc = VocoderConfig(audio=cfg.audio, causal=True, lookahead=la, dim=32, intermediate_dim=64, n_layers=8)
    return cfg, vc, fact, Generator(vc, in_dim=input_dim(fact)).eval()


@pytest.mark.parametrize("la, expected", [(0, 120.0), (4, 200.0), (8, 280.0)])
def test_formula_matches_measurement(la, expected):
    cfg, vc, fact, gen = _setup(la)
    assert algorithmic_latency_ms(cfg, vc) == pytest.approx(expected)
    mel_fn = LogMelSpectrogram(cfg.audio)
    ref = mel_fn(0.1 * torch.randn(1, 48_000, generator=torch.Generator().manual_seed(1)))
    got = measure_ms(fact, gen, mel_fn, ref, n_samples=480 * 120, probe_from=480 * 60, probe_len=1920, step=60)
    # The formula is an upper bound. On a 60-sample (2.5 ms) probe grid, with
    # the Hann window exactly 0 at its edges, the worst case found sits within
    # 7.5 ms below it, never above.
    assert expected - 7.5 - 1e-6 <= got <= expected + 1e-6


def test_noncausal_encoder_has_no_bounded_latency():
    cfg, vc, _, _ = _setup(4)
    cfg.encoder = dataclasses.replace(cfg.encoder, causal=False)
    assert algorithmic_latency_ms(cfg, vc) is None
