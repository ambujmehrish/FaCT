"""Algorithmic latency of a FaCT encoder + stage-D generator pair.

Algorithmic latency is the worst-case time between an input sample arriving
and the last output sample that depends on it being computable. Compute time
is excluded and measured separately. With a 24 kHz input, hop 480 (20 ms
frames) and frame_stack 4 (80 ms tokens), the terms are:

  token granularity   (frame_stack - 1) frames: a token waits for the rest of
                      its 4 frames (60 ms)
  chunk lookahead     (chunk_tokens - 1) tokens (0 for a strictly causal
                      encoder)
  STFT analysis       half the analysis window ahead of a frame's centre
                      (win_length / 2 = 480 samples = 20 ms)
  generator lookahead VocoderConfig.lookahead frames of convolution
                      lookahead, plus 2 frames from the iSTFT overlap-add
                      (n_fft = 4 hops)

For a causal encoder this gives 120 / 200 / 280 ms with generator lookahead
0 / 4 / 8 frames (la0 / la4 / la8). measure_ms() checks the formula by
perturbation; tests/test_latency.py compares the two.
"""

from __future__ import annotations

import torch


def algorithmic_latency_ms(fact_cfg, voc_cfg) -> float | None:
    """None when the encoder is not causal (it needs the whole utterance)."""
    enc, a = fact_cfg.encoder, fact_cfg.audio
    if not enc.causal or enc.ssl_model:
        return None
    if not voc_cfg.causal:
        return None
    frame_ms = 1000.0 * a.hop_length / a.sample_rate
    token = (enc.frame_stack - 1) * frame_ms
    chunk = (enc.chunk_tokens - 1) * enc.frame_stack * frame_ms
    stft = 1000.0 * (a.win_length // 2) / a.sample_rate
    istft = (voc_cfg.n_fft // voc_cfg.audio.hop_length - 2) * frame_ms
    gen = voc_cfg.lookahead * frame_ms + istft
    return token + chunk + stft + gen


@torch.no_grad()
def measure_ms(fact, gen, mel_fn, ref_mel: torch.Tensor, n_samples: int, probe_from: int,
               probe_len: int, step: int = 60) -> float:
    """Worst-case lookahead found by perturbing one input sample at a time
    and finding the earliest output sample that changes (ms). ref_mel: a
    fixed speaker reference, so that the global speaker vector does not
    depend on the probed audio."""
    from ..train.stage_d_wave_decoder import fact_condition
    s = fact.cfg.encoder.frame_stack
    sr = fact.cfg.audio.sample_rate
    wav = 0.1 * torch.randn(1, n_samples, generator=torch.Generator().manual_seed(0))

    def run(w):
        mel = mel_fn(w)
        mel = mel[:, : mel.shape[1] // s * s]
        torch.manual_seed(5)  # the capped residual adds channel noise
        return gen(fact_condition(fact, mel, ref_mel=ref_mel))

    y = run(wav)
    worst = 0
    for tau in range(probe_from, probe_from + probe_len, step):
        w2 = wav.clone()
        w2[0, tau] += 1.0
        changed = ((run(w2) - y).abs() > 0).nonzero()
        if len(changed):
            worst = max(worst, tau - int(changed[:, 1].min()))
    return 1000.0 * worst / sr
