"""Frozen self-supervised speech features for the encoder input (#4).

WavLM-large (microsoft/wavlm-large), truncated to its first `layer`
transformer layers - FocalCodec's choice (layer 6) - and run under
no_grad. It is not part of the FaCT checkpoint: the trainer and the
evaluation scripts hold it next to the model.

Frame alignment with FaCT's mel (24 kHz, hop 480, center=True): mel frame
k is centred at k x 20 ms. WavLM's conv front end has a 400-sample
receptive field and a 320-sample stride at 16 kHz, so frame j covers
samples [320 j, 320 j + 400) and is centred at 320 j + 200. Padding 200
samples on each side centres WavLM frame j at 320 j - the same instant as
mel frame j - and yields N / 320 + 1 frames for N samples, exactly the mel
frame count when N = (T_mel - 1) x 320.
"""

from __future__ import annotations

import torch
from torch import nn

SSL_SR = 16_000
HOP = 320
PAD = 200


def ssl_samples(n_mel_frames: int) -> int:
    """16 kHz samples whose SSL frames align 1:1 with n_mel_frames mel frames."""
    return (n_mel_frames - 1) * HOP


class SSLFeatures(nn.Module):
    def __init__(self, model_id: str = "microsoft/wavlm-large", layer: int = 6,
                 model: nn.Module | None = None):
        """model: an already-built WavLMModel (tests); else from_pretrained."""
        super().__init__()
        from transformers import WavLMModel
        m = model if model is not None else WavLMModel.from_pretrained(model_id)
        if not 1 <= layer <= len(m.encoder.layers):
            raise ValueError(f"layer {layer} outside 1..{len(m.encoder.layers)}")
        m.encoder.layers = m.encoder.layers[:layer]  # later layers are never used
        m.config.num_hidden_layers = layer
        self.model = m.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim = m.config.hidden_size
        self.layer = layer

    def train(self, mode: bool = True):  # always frozen / eval
        return super().train(False)

    @torch.no_grad()
    def forward(self, wav16: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """wav16: (B, N) at 16 kHz, zero-padded; lengths: (B,) valid samples.
        Returns (B, N // 320 + 1, dim) features of the last kept layer."""
        b, n = wav16.shape
        if lengths is None:
            lengths = torch.full((b,), n, dtype=torch.long, device=wav16.device)
        valid = (torch.arange(n, device=wav16.device)[None] < lengths[:, None]).float()
        # Per-utterance zero-mean / unit-variance over valid samples (the
        # normalization WavLM-large was trained with).
        cnt = valid.sum(1, keepdim=True).clamp_min(1)
        mean = (wav16 * valid).sum(1, keepdim=True) / cnt
        var = (((wav16 - mean) * valid) ** 2).sum(1, keepdim=True) / cnt
        x = (wav16 - mean) / (var + 1e-7).sqrt() * valid
        # Each item: PAD zeros | its N_i samples | PAD zeros, all marked
        # valid, so every item is framed exactly as it would be alone.
        x = torch.nn.functional.pad(x, (PAD, PAD))
        mask = (torch.arange(n + 2 * PAD, device=wav16.device)[None]
                < (lengths + 2 * PAD)[:, None]).long()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=wav16.is_cuda):
            out = self.model(x, attention_mask=mask).last_hidden_state
        return out.float()


def features_for(ssl: "SSLFeatures | None", wav: torch.Tensor, sr: int,
                 n_mel_frames: int, device) -> torch.Tensor | None:
    """SSL features for one utterance (wav: (N,) at sr), aligned with its
    first n_mel_frames mel frames; None when the model takes no SSL input."""
    if ssl is None:
        return None
    return batch_features_for(ssl, wav.detach().float().cpu()[None], sr, n_mel_frames, device)


def batch_features_for(ssl: "SSLFeatures | None", wav: torch.Tensor, sr: int,
                       n_mel_frames: int, device) -> torch.Tensor | None:
    """features_for for a batch of equal-length clips (wav: (B, N) at sr),
    e.g. stage D's training segments; resamples on wav's device."""
    if ssl is None:
        return None
    from ..eval.metrics_audio import resample
    w = resample(wav.detach().float(), sr, SSL_SR)
    n = ssl_samples(n_mel_frames)
    w = torch.nn.functional.pad(w[:, :n], (0, max(0, n - w.shape[1])))
    return ssl(w.to(device))[:, :n_mel_frames]


def ssl_for_config(cfg, device) -> "SSLFeatures | None":
    """The frozen extractor a FaCT config needs (None for mel-only models)."""
    if not cfg.encoder.ssl_model:
        return None
    m = SSLFeatures(cfg.encoder.ssl_model, cfg.encoder.ssl_layer).to(device)
    if m.dim != cfg.encoder.ssl_dim:
        raise ValueError(f"{cfg.encoder.ssl_model}: {m.dim}-dim features, config says {cfg.encoder.ssl_dim}")
    return m
