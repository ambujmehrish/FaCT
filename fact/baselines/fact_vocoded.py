"""FaCT (+ stage-C vocoder) as an entry of the waveform baseline harness.

Wrapping FaCT behind the same reconstruct(wav) -> wav interface as Mimi /
DAC / X-codec2 means scripts/eval_baselines.py scores it with exactly the
same PESQ / STOI / WER / F0 / mel code - no FaCT-specific metric path.

    fact:ckpt=<step.pt>:vocoder=<voc.pt>:view=full:label=fact_main_full
    fact_vocoder:vocoder=<voc.pt>:label=vocoder_copysyn   (vocoder ceiling)
    ...:spk_ref_seconds=3   speaker vector from the first 3 s only (streaming-valid)

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


class _SpeakerRef:
    """Speaker-reference protocol for FaCT entries of the harness.

    full       the utterance itself (not streaming-valid; the speaker vector
               can carry utterance-specific detail - an upper bound)
    other      a different utterance of the same speaker from the eval set
               (enrollment; the fair protocol vs codecs without side info)
    prefix:<s> the utterance's first <s> seconds
    The harness passes the eval set (set_eval_set) and each utterance's key.
    """

    takes_key = True

    def _init_ref(self, spk_ref: str, spk_ref_seconds: float):
        if float(spk_ref_seconds) > 0:
            spk_ref = f"prefix:{float(spk_ref_seconds)}"
        mode, _, arg = str(spk_ref).partition(":")
        if mode not in ("full", "other", "prefix") or (mode == "prefix") != bool(arg):
            raise ValueError(f"spk_ref {spk_ref!r}: use full, other or prefix:<seconds>")
        self.spk_mode, self.spk_arg = mode, arg
        self.partners, self.by_key = {}, {}

    def set_eval_set(self, wavs) -> None:
        if self.spk_mode == "other":
            from ..eval.eval_set import reference_partners
            self.partners = reference_partners([w[0] for w in wavs], skip_singletons=True)
            self.by_key = {rel: (w, sr) for rel, w, sr in wavs}

    def _ref_mel(self, mel: torch.Tensor, key) -> torch.Tensor:
        if self.spk_mode == "full":
            return mel
        if self.spk_mode == "prefix":
            a = self.mel_fn.cfg
            n = int(float(self.spk_arg) * a.sample_rate / a.hop_length)
            return mel[:, : max(1, n)]
        if key not in self.partners:
            raise KeyError(f"{key}: no same-speaker partner (run the eval with --partner-subset)")
        from ..eval.metrics_audio import resample
        w, sr = self.by_key[self.partners[key]]
        return self.mel_fn(resample(w, sr, self.sample_rate).to(self.device).unsqueeze(0))


class FaCTVocoded(_SpeakerRef):
    name = "fact"

    def __init__(self, ckpt: str, vocoder: str, view: str = "full", nfe: int = 32,
                 sampler: str = "fm", seed: int = 0, spk_ref_seconds: float = 0.0,
                 temperature: float = 1.0, cfg_scale: float = 1.0, spk_ref: str = "full",
                 device: str = "cuda"):
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
        from ..modules.ssl import ssl_for_config
        self.ssl_fn = ssl_for_config(cfg, device)  # None for mel-only models
        self.view, self.nfe, self.shortcut = view, nfe, sampler == "shortcut"
        self.seed, self.n_done, self.device = seed, 0, device
        self.temperature, self.cfg_scale = float(temperature), float(cfg_scale)
        self._init_ref(spk_ref, spk_ref_seconds)
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

    def _ssl(self, wav: torch.Tensor, n_frames: int):
        from ..modules.ssl import features_for
        return features_for(self.ssl_fn, wav[0], self.sample_rate, n_frames, self.device)

    @torch.no_grad()
    def tokenize(self, wav: torch.Tensor):
        mel = self._mel(wav)
        out = self.model.encode_mel(mel, self._ssl(wav, mel.shape[1]))
        return [x for x in (out.content_indices, out.prosody_indices) if x is not None]

    @torch.no_grad()
    def reconstruct(self, wav: torch.Tensor, key=None) -> torch.Tensor:
        mel = self._mel(wav)
        ref = self._ref_mel(mel, key)
        ssl = self._ssl(wav, mel.shape[1])
        torch.manual_seed(self.seed + self.n_done)
        self.n_done += 1
        if self.view == "full":
            hyp = self.model.reconstruct(mel, ref_mel=ref, nfe=self.nfe, shortcut=self.shortcut,
                                         ssl=ssl, temperature=self.temperature,
                                         cfg_scale=self.cfg_scale)
        else:
            out = self.model.encode_mel(mel, ssl)
            residual = None
            if self.view == "tokens_prior" and self.model.bottleneck.residual_up is not None:
                residual = torch.zeros(*out.content_indices.shape,
                                       self.model.bottleneck.residual_dim, device=self.device)
            hyp = self.model.detokenize(out.content_indices, out.prosody_indices, ref_mel=ref,
                                        residual=residual, nfe=self.nfe, shortcut=self.shortcut,
                                        temperature=self.temperature, cfg_scale=self.cfg_scale)
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
                nfe: int = 32, sampler: str = "fm", seed: int = 0,
                spk_ref_seconds: float = 0.0, temperature: float = 1.0,
                cfg_scale: float = 1.0, spk_ref: str = "full") -> FaCTVocoded:
    if not ckpt or not vocoder:
        raise ValueError("fact baseline needs ckpt=<step.pt> and vocoder=<voc.pt>")
    return FaCTVocoded(ckpt, vocoder, view=view, nfe=nfe, sampler=sampler, seed=seed,
                       spk_ref_seconds=spk_ref_seconds, temperature=temperature,
                       cfg_scale=cfg_scale, spk_ref=spk_ref, device=device)


@register("fact_vocoder")
def _build_copysyn(device: str = "cuda", vocoder: str = "") -> VocoderCopySynthesis:
    if not vocoder:
        raise ValueError("fact_vocoder needs vocoder=<voc.pt>")
    return VocoderCopySynthesis(vocoder, device=device)


class FaCTWave(_SpeakerRef):
    """FaCT tokens -> stage-D waveform decoder (#2): deterministic, no
    flow-matching sampling and no mel vocoder."""

    name = "fact_wave"

    def __init__(self, decoder: str, view: str = "full", ckpt: str = "", spk_ref: str = "full",
                 device: str = "cuda"):
        from ..eval.bitrate import fact_bitrate
        from ..modules.mel import LogMelSpectrogram
        from ..train.stage_d_wave_decoder import load_fact_frozen, load_wave_decoder
        if view not in ("full", "discrete_only"):
            raise ValueError("fact_wave view must be full or discrete_only")
        self.gen, meta = load_wave_decoder(decoder, device)
        self.fact = load_fact_frozen(ckpt or meta["fact_ckpt"], device)
        bn = self.fact.bottleneck
        if view == "discrete_only" and bn.residual_up is not None and not meta["residual_drop"]:
            raise ValueError("this decoder was never trained without the residual")
        self.view, self.device = view, device
        self._init_ref(spk_ref, 0.0)
        self.mel_fn = LogMelSpectrogram(self.fact.cfg.audio).to(device)
        self.sample_rate = self.fact.cfg.audio.sample_rate
        from ..modules.ssl import ssl_for_config
        self.ssl_fn = ssl_for_config(self.fact.cfg, device)  # None for mel-only models
        br = fact_bitrate(self.fact)
        self.frame_rate_hz = br["token_rate_hz"]
        res = br.get("residual_bps")
        self.discrete_bps = br["discrete_bps"] + ((res or 0.0) if view == "full" else 0.0)
        self.vocab_sizes = [v for v in (bn.content_codebook_size, bn.prosody_codebook_size) if v]

    def _mel(self, wav):
        s = self.fact.cfg.encoder.frame_stack
        mel = self.mel_fn(wav.to(self.device))
        return mel[:, : mel.shape[1] // s * s]

    def _ssl(self, wav, n_frames):
        from ..modules.ssl import features_for
        return features_for(self.ssl_fn, wav[0], self.sample_rate, n_frames, self.device)

    @torch.no_grad()
    def tokenize(self, wav):
        mel = self._mel(wav)
        out = self.fact.encode_mel(mel, self._ssl(wav, mel.shape[1]))
        return [x for x in (out.content_indices, out.prosody_indices) if x is not None]

    @torch.no_grad()
    def reconstruct(self, wav, key=None):
        from ..train.stage_d_wave_decoder import fact_condition
        mel = self._mel(wav)
        drop = torch.full((mel.shape[0],), self.view != "full", device=self.device)
        return self.gen(fact_condition(self.fact, mel, drop, ssl=self._ssl(wav, mel.shape[1]),
                                       ref_mel=self._ref_mel(mel, key)))


@register("fact_wave")
def _build_wave(device: str = "cuda", decoder: str = "", view: str = "full",
                ckpt: str = "", spk_ref: str = "full") -> FaCTWave:
    if not decoder:
        raise ValueError("fact_wave needs decoder=<stage-D step.pt>")
    return FaCTWave(decoder, view=view, ckpt=ckpt, spk_ref=spk_ref, device=device)
