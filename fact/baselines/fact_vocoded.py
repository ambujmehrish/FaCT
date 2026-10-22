"""FaCT (+ stage-C vocoder) as an entry of the waveform baseline harness.

Wrapping FaCT behind the same reconstruct(wav) -> wav interface as Mimi /
DAC / X-codec2 means scripts/eval_baselines.py scores it with exactly the
same PESQ / STOI / WER / F0 / mel code - no FaCT-specific metric path.

    fact:ckpt=<step.pt>:vocoder=<voc.pt>:view=full:label=fact_main_full
    fact_vocoder:vocoder=<voc.pt>:label=vocoder_copysyn   (vocoder ceiling)

Views match scripts/eval_fact_recon.py:
  full           content + prosody tokens + residual channel
  tokens_prior   tokens; residual = message 0 through the channel
  discrete_only  tokens; residual removed (residual-dropout models)
Sampling: plain flow matching (stage-A checkpoints) with `nfe` Euler steps;
sampler=shortcut for stage-B checkpoints. The decoder noise is re-seeded
per utterance (seed + utterance index), as in eval_fact_recon.py.
discrete_bps reports what the view transmits: tokens, plus the residual's
capacity bound for `full`.
"""

from __future__ import annotations

import torch

from .base import register

VIEWS = ("full", "tokens_prior", "discrete_only")
MEL_FIELDS = ("sample_rate", "n_mels", "n_fft", "hop_length", "win_length", "fmin", "fmax")


def _same_mel_space(a, b) -> list[str]:
    return [f for f in MEL_FIELDS if getattr(a, f) != getattr(b, f)]


class FaCTVocoded:
    name = "fact"

    def __init__(self, ckpt: str, vocoder: str, view: str = "full", nfe: int = 32,
                 sampler: str = "fm", seed: int = 0, device: str = "cuda"):
        from ..config import FaCTConfig
        from ..eval.bitrate import fact_bitrate
        from ..model import FaCT
        from ..modules.mel import LogMelSpectrogram
        from ..train.stage_c_vocoder import load_generator
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}, got {view!r}")
        if sampler not in ("fm", "shortcut"):
            raise ValueError(f"sampler must be fm or shortcut, got {sampler!r}")
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
        cfg = FaCTConfig.from_dict(ck["config"])
        self.model = FaCT(cfg)
        self.model.load_state_dict(ck["model"])
        self.model = self.model.to(device).eval()
        self.vocoder = load_generator(vocoder, device)
        bad = _same_mel_space(cfg.audio, self.vocoder.cfg.audio)
        if bad:
            raise ValueError(f"FaCT and vocoder mel spaces differ in {bad}")
        bn = self.model.bottleneck
        if view != "full" and bn.content_codebook_size is None:
            raise ValueError("token views need a discrete content stream")
        self.mel_fn = LogMelSpectrogram(cfg.audio).to(device)
        self.view, self.nfe, self.shortcut = view, nfe, sampler == "shortcut"
        self.seed, self.n_done, self.device = seed, 0, device
        self.sample_rate = cfg.audio.sample_rate
        br = fact_bitrate(self.model)
        self.frame_rate_hz = br["token_rate_hz"]
        res = br.get("residual_bps")
        if view == "full" and bn.residual_down is not None and res is None:
            self.discrete_bps = None  # uncapped (KL) residual: no bitrate bound
        else:
            self.discrete_bps = br["discrete_bps"] + ((res or 0.0) if view == "full" else 0.0)
        self.vocab_sizes = [v for v in (bn.content_codebook_size, bn.prosody_codebook_size) if v]

    def _mel(self, wav: torch.Tensor) -> torch.Tensor:
        s = self.model.cfg.encoder.frame_stack
        mel = self.mel_fn(wav.to(self.device))
        mel = mel[:, : (mel.shape[1] // s) * s]
        if mel.shape[1] == 0:
            raise ValueError("utterance shorter than one token")
        return mel

    @torch.no_grad()
    def tokenize(self, wav: torch.Tensor):
        out = self.model.encode_mel(self._mel(wav))
        return [x for x in (out.content_indices, out.prosody_indices) if x is not None]

    @torch.no_grad()
    def reconstruct(self, wav: torch.Tensor) -> torch.Tensor:
        mel = self._mel(wav)
        torch.manual_seed(self.seed + self.n_done)
        self.n_done += 1
        if self.view == "full":
            hyp = self.model.reconstruct(mel, nfe=self.nfe, shortcut=self.shortcut)
        else:
            out = self.model.encode_mel(mel)
            residual = None
            if self.view == "tokens_prior" and self.model.bottleneck.residual_up is not None:
                residual = torch.zeros(*out.content_indices.shape,
                                       self.model.bottleneck.residual_dim, device=self.device)
            hyp = self.model.detokenize(out.content_indices, out.prosody_indices, ref_mel=mel,
                                        residual=residual, nfe=self.nfe, shortcut=self.shortcut)
        return self.vocoder(hyp)


class VocoderCopySynthesis:
    """Ground-truth mel -> stage-C vocoder: the ceiling for every FaCT row."""

    name = "fact_vocoder"
    frame_rate_hz = None
    discrete_bps = None
    vocab_sizes = None

    def __init__(self, vocoder: str, device: str = "cuda"):
        from ..modules.mel import LogMelSpectrogram
        from ..train.stage_c_vocoder import load_generator
        self.vocoder = load_generator(vocoder, device)
        self.mel_fn = LogMelSpectrogram(self.vocoder.cfg.audio).to(device)
        self.sample_rate = self.vocoder.cfg.audio.sample_rate
        self.frame_rate_hz = self.sample_rate / self.vocoder.cfg.audio.hop_length
        self.device = device

    def tokenize(self, wav):
        return None

    @torch.no_grad()
    def reconstruct(self, wav: torch.Tensor) -> torch.Tensor:
        return self.vocoder(self.mel_fn(wav.to(self.device)))


@register("fact")
def _build_fact(device: str = "cuda", ckpt: str = "", vocoder: str = "", view: str = "full",
                nfe: int = 32, sampler: str = "fm", seed: int = 0) -> FaCTVocoded:
    if not ckpt or not vocoder:
        raise ValueError("fact baseline needs ckpt=<step.pt> and vocoder=<voc.pt>")
    return FaCTVocoded(ckpt, vocoder, view=view, nfe=nfe, sampler=sampler, seed=seed,
                       device=device)


@register("fact_vocoder")
def _build_copysyn(device: str = "cuda", vocoder: str = "") -> VocoderCopySynthesis:
    if not vocoder:
        raise ValueError("fact_vocoder needs vocoder=<voc.pt>")
    return VocoderCopySynthesis(vocoder, device=device)
