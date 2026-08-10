"""Configuration dataclasses for FaCT.

All sizes default to the paper-scale model described in the research plan
(~100M encoder, ~200M decoder, 12.5 Hz token rate). Tests and smoke runs
override with :func:`tiny_config`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class AudioConfig:
    sample_rate: int = 24_000
    n_mels: int = 100
    n_fft: int = 1024
    hop_length: int = 480  # 24 kHz / 480 = 50 Hz mel frame rate
    win_length: int = 960
    fmin: float = 0.0
    fmax: float = 12_000.0
    # Global log-mel normalization (flow matching wants ~unit-scale targets).
    mel_mean: float = -4.0
    mel_std: float = 3.0


@dataclass
class EncoderConfig:
    """Causal transformer encoder: mel @ 50 Hz -> latents @ 12.5 Hz."""

    dim: int = 768
    n_layers_pre: int = 6   # layers at 50 Hz, before the 4x frame stack
    n_layers_post: int = 10  # layers at 12.5 Hz, after the stack
    n_heads: int = 12
    ffn_mult: float = 4.0
    dropout: float = 0.0
    frame_stack: int = 4  # 50 Hz -> 12.5 Hz
    conv_kernel: int = 5  # causal depthwise conv in the frontend
    causal: bool = True   # False = bidirectional (streamability ablation)


@dataclass
class BottleneckConfig:
    """Factorized semi-discrete bottleneck (and its matched baselines).

    variant:
      "factorized" (FaCT): content FSQ (8*8*8*4*4 = 8192 = 2^13 codes)
        + prosody FSQ (5^4 = 625 ~ 2^9.3 codes) + optional continuous residual.
      "single_fsq": one FSQ head with `single_levels`. Default is the union
        of the content and prosody lattices - exactly matched bits (2^22.3)
        and dims (9) - isolating factorization (ablation #1 / RQ2).
      "vae": pure-continuous KL-regularized latent of `vae_dim` dims.
        Default 25 = content 5 + prosody 4 + residual 16 dims - the matched
        continuous baseline (RQ1/RQ3). `residual_kl_weight` is its beta.
    """

    variant: str = "factorized"  # "factorized" | "single_fsq" | "vae" | "rvq"
    content_levels: tuple[int, ...] = (8, 8, 8, 4, 4)
    prosody_levels: tuple[int, ...] = (5, 5, 5, 5)
    residual_dim: int = 16  # 0 disables the residual channel
    residual_kl_weight: float = 1e-2
    single_levels: tuple[int, ...] = (8, 8, 8, 4, 4, 5, 5, 5, 5)
    vae_dim: int = 25
    # "rvq" (quantizer-swap ablation): content head uses EMA residual VQ at
    # matched bits (91*91 = 8281 ~ 2^13.02 vs FSQ's 2^13); prosody FSQ,
    # residual, and all losses stay identical, isolating the quantizer.
    rvq_codebook_sizes: tuple[int, ...] = (91, 91)
    rvq_dim: int = 8
    rvq_decay: float = 0.99
    rvq_commitment: float = 0.25


@dataclass
class CTCConfig:
    """CTC-on-quantized-content-tokens semantic grounding (SiTok mechanism).

    upsample: CTC logit positions emitted per 12.5 Hz token. Byte-level
    English text runs ~12-18 bytes/s - AT or ABOVE the token rate - so
    without upsampling CTC is infeasible (T < S) for many real utterances
    and zero_infinity silently zeroes the loss. upsample=2 gives 25
    positions/s. The training loop logs `ctc_infeasible` (fraction of the
    batch with S > T*upsample); keep it near zero.
    """

    n_layers: int = 4
    n_heads: int = 8
    dim: int = 512
    vocab_size: int = 257  # 256 byte values (shifted +1) + blank at 0
    blank_id: int = 0
    weight: float = 1.0
    upsample: int = 2


@dataclass
class ProsodyConfig:
    """Supervision targets for the prosody head (cheap, auto-extractable)."""

    f0_bins: int = 64          # quantized log-F0 classes (+1 unvoiced bin)
    f0_min: float = 50.0
    f0_max: float = 550.0
    f0_weight: float = 1.0
    energy_weight: float = 1.0
    predictor_dim: int = 256


@dataclass
class LeakageConfig:
    """Cross-leakage penalties via gradient-reversal probes (DSA-style)."""

    grl_lambda: float = 0.5
    content_to_f0_weight: float = 0.5   # content must NOT predict F0
    prosody_to_text_weight: float = 0.5  # prosody must NOT predict text
    probe_dim: int = 256


@dataclass
class SpeakerConfig:
    """Global reference-encoder speaker conditioning (lives in neither stream)."""

    dim: int = 256
    n_layers: int = 4


@dataclass
class DecoderConfig:
    """Causal (block-wise) shortcut flow-matching mel decoder."""

    dim: int = 1024
    n_layers: int = 12
    n_heads: int = 16
    ffn_mult: float = 4.0
    dropout: float = 0.0
    block_size: int = 8       # attention block, in 50 Hz mel frames
    causal: bool = True       # False = full attention (streamability ablation)
    cfg_dropout: float = 0.1  # condition dropout for classifier-free guidance
    max_shortcut_log2: int = 7  # supports step sizes 1/128 ... 1


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 5_000
    max_steps: int = 400_000
    batch_seconds: float = 320.0
    grad_accum: int = 1  # raise on 4x64GB nodes to keep the effective batch
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    seed: int = 1234
    log_every: int = 50
    ckpt_every: int = 5_000


@dataclass
class FaCTConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    bottleneck: BottleneckConfig = field(default_factory=BottleneckConfig)
    ctc: CTCConfig = field(default_factory=CTCConfig)
    prosody: ProsodyConfig = field(default_factory=ProsodyConfig)
    leakage: LeakageConfig = field(default_factory=LeakageConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def token_rate_hz(self) -> float:
        mel_rate = self.audio.sample_rate / self.audio.hop_length
        return mel_rate / self.encoder.frame_stack

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FaCTConfig":
        def build(klass, sub):
            fields = {f.name: f for f in dataclasses.fields(klass)}
            kwargs = {}
            for k, v in sub.items():
                if k not in fields:
                    raise KeyError(f"Unknown config key {k!r} for {klass.__name__}")
                if isinstance(v, list):
                    v = tuple(v)
                kwargs[k] = v
            return klass(**kwargs)

        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name in d:
                kwargs[f.name] = build(f.default_factory, d[f.name])
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str) -> "FaCTConfig":
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def save_yaml(self, path: str) -> None:
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)


def tiny_config() -> FaCTConfig:
    """A CPU-runnable configuration for tests and smoke training."""
    cfg = FaCTConfig()
    cfg.audio = AudioConfig(sample_rate=16_000, n_mels=40, n_fft=512, hop_length=320, win_length=512, fmax=8_000.0)
    cfg.encoder = EncoderConfig(dim=64, n_layers_pre=1, n_layers_post=1, n_heads=4, frame_stack=4, conv_kernel=3)
    cfg.bottleneck = BottleneckConfig(content_levels=(5, 5, 5), prosody_levels=(4, 4), residual_dim=4)
    cfg.ctc = CTCConfig(n_layers=1, n_heads=4, dim=64, vocab_size=32)
    cfg.prosody = ProsodyConfig(f0_bins=16, predictor_dim=32)
    cfg.leakage = LeakageConfig(probe_dim=32)
    cfg.speaker = SpeakerConfig(dim=32, n_layers=1)
    cfg.decoder = DecoderConfig(dim=64, n_layers=2, n_heads=4, block_size=4, max_shortcut_log2=4)
    cfg.train = TrainConfig(warmup_steps=10, max_steps=50, batch_seconds=4.0, log_every=5, ckpt_every=1_000)
    return cfg
