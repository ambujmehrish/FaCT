"""Pin the metric wiring to the reference implementations: identity and
known-degradation behavior. These are the checks that make cluster numbers
trustworthy."""

import math

import numpy as np
import pytest
import torch

from fact.eval.metrics_audio import UtteranceMetrics, aggregate, si_sdr
from fact.eval.wer import normalized_wer

pesq = pytest.importorskip("pesq", reason="pesq not installed")
pystoi = pytest.importorskip("pystoi", reason="pystoi not installed")

from fact.eval.metrics_audio import pesq_wb, stoi  # noqa: E402


def speechlike(sr: int, seconds: float = 2.0, f0: float = 150.0) -> torch.Tensor:
    t = torch.arange(int(sr * seconds)) / sr
    wav = torch.zeros_like(t)
    for h in range(1, 6):
        wav += torch.sin(2 * math.pi * h * f0 * t) / h
    env = 0.5 * (1 + torch.sin(2 * math.pi * 3.0 * t))  # syllable-rate modulation
    return 0.5 * wav * env / wav.abs().max()


def test_pesq_identity_is_max():
    wav = speechlike(16_000)
    score = pesq_wb(wav, wav, 16_000)
    assert score > 4.5  # P.862.2 identity ceiling is 4.64


def test_stoi_identity_is_one():
    wav = speechlike(16_000)
    assert stoi(wav, wav, 16_000) > 0.99


def test_metrics_degrade_with_noise():
    wav = speechlike(16_000)
    noisy = wav + 0.1 * torch.randn_like(wav)
    assert pesq_wb(wav, noisy, 16_000) < pesq_wb(wav, wav, 16_000) - 0.5
    assert stoi(wav, noisy, 16_000) < 1.0
    assert si_sdr(wav, noisy) < si_sdr(wav, wav) - 10


def test_metrics_handle_resample_path():
    """24 kHz input goes through the 16 kHz resample path without error."""
    wav = speechlike(24_000)
    assert pesq_wb(wav, wav, 24_000) > 4.4


def test_si_sdr_scale_invariance():
    wav = speechlike(16_000)
    noisy = wav + 0.05 * torch.randn_like(wav)
    # Rescaling the estimate must not change SI-SDR (that's the "SI").
    assert abs(si_sdr(wav, 0.5 * noisy) - si_sdr(wav, noisy)) < 1e-3


def test_aggregate_skips_failures():
    items = [
        UtteranceMetrics(pesq_wb=2.0, stoi=0.9),
        UtteranceMetrics(pesq_wb=None, stoi=0.8, extras={"wer": 0.1}),
        UtteranceMetrics(pesq_wb=float("nan"), stoi=0.7),
    ]
    agg = aggregate(items)
    assert agg["pesq_wb"]["n"] == 1
    assert agg["stoi"]["n"] == 3
    assert agg["wer"]["n"] == 1


def test_wer():
    assert normalized_wer("hello world", "hello world") == 0.0
    assert normalized_wer("Hello, World!", "hello world") == 0.0  # normalization
    assert normalized_wer("a b c d", "a x c d") == 0.25
    assert normalized_wer("", "anything") == 1.0
