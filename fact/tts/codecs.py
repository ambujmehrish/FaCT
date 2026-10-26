"""Codec adapters for TTS evaluation: prompt audio -> codes (S, T) and
generated codes -> waveform, for the codecs the LMs were trained on.

FaCT: content + prosody indices from the same frozen model the training
tokens came from (fact.tts.tokenize), rendered by the stage-D waveform
decoder from the indices alone (tokens-only view; the residual is never
generated) with the speaker vector of the prompt. Mimi: its own decoder.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


def _same_file(a: str, b: str) -> bool:
    return os.path.realpath(a) == os.path.realpath(b)


class FaCTTTSCodec:
    def __init__(self, codec_meta: dict, wave_decoder: str, device: str):
        from ..modules.mel import LogMelSpectrogram
        from ..train.stage_d_wave_decoder import load_fact_frozen, load_wave_decoder
        self.gen, meta = load_wave_decoder(wave_decoder, device)
        if not _same_file(meta["fact_ckpt"], codec_meta["fact_ckpt"]):
            raise SystemExit(f"stage-D decoder was trained on {meta['fact_ckpt']}, the LM's tokens "
                             f"come from {codec_meta['fact_ckpt']}")
        self.fact = load_fact_frozen(codec_meta["fact_ckpt"], device)
        bn = self.fact.bottleneck
        if bn.residual_up is not None and not meta["residual_drop"]:
            raise SystemExit("this stage-D decoder was never trained without the residual")
        self.sample_rate = self.fact.cfg.audio.sample_rate
        self.s = self.fact.cfg.encoder.frame_stack
        self.mel_fn = LogMelSpectrogram(self.fact.cfg.audio).to(device)
        self.device = device

    def _mel(self, wav: torch.Tensor) -> torch.Tensor:
        m = self.mel_fn(wav.to(self.device)[None])
        return m[:, : m.shape[1] // self.s * self.s]

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        out = self.fact.encode_mel(self._mel(wav))
        return torch.stack([out.content_indices[0], out.prosody_indices[0]]).cpu()

    @torch.no_grad()
    def decode(self, codes: torch.Tensor, prompt_wav: torch.Tensor) -> torch.Tensor:
        c = codes.to(self.device).long()
        cond = self.fact.bottleneck.embed_indices(c[0][None], c[1][None], None)
        cond = cond.repeat_interleave(self.s, dim=1)
        spk = self.fact.speaker_encoder(self.fact._norm_mel(self._mel(prompt_wav)))
        x = torch.cat([cond, spk[:, None].expand(-1, cond.shape[1], -1)], dim=-1)
        return self.gen(x)[0].float().cpu()


class MimiTTSCodec:
    def __init__(self, codec_meta: dict, device: str):
        from ..baselines.mimi import MimiTokenStreams
        self.m = MimiTokenStreams(n_streams=int(codec_meta["mimi_streams"]), device=device)
        self.sample_rate = self.m.sample_rate

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        return self.m._encode(wav[None])[0].cpu()

    @torch.no_grad()
    def decode(self, codes: torch.Tensor, prompt_wav: torch.Tensor) -> torch.Tensor:
        audio = self.m.model.decode(codes[None].to(self.m.device).long()).audio_values
        return audio[0, 0].float().cpu()


def build_codec(codec_meta: dict, device: str, wave_decoder: str | None = None):
    if codec_meta["codec"] == "fact":
        if not wave_decoder or not Path(wave_decoder).is_file():
            raise SystemExit("FaCT TTS eval needs --wave-decoder (stage-D step.pt)")
        return FaCTTTSCodec(codec_meta, wave_decoder, device)
    if codec_meta["codec"] == "mimi":
        return MimiTTSCodec(codec_meta, device)
    raise SystemExit(f"unknown codec {codec_meta['codec']!r}")
