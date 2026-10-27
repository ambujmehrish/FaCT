"""Stage-C vocoder: FaCT log-mel (24 kHz, hop 480, 100 bins) -> waveform.

Architecture and losses follow Vocos (Siuzdak, arXiv:2306.00814): a
ConvNeXt backbone running at the mel frame rate and an iSTFT head that
predicts STFT magnitude and phase per frame, inverted with torch.istft, so
there is no upsampling stack. Trained as a GAN against the HiFi-GAN
multi-period discriminator and the UnivNet multi-resolution spectrogram
discriminator, with hinge losses, feature matching and mel L1.

Only the feature layout differs from the public 24 kHz Vocos (hop 480
here, 256 there), which is why it is retrained on FaCT's mel. The iSTFT
uses n_fft = 4 x hop, the same ratio as Vocos (1024 / 256).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from ..config import AudioConfig
from ..modules.mel import LogMelSpectrogram


@dataclass
class VocoderConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)  # the input mel space
    dim: int = 512
    intermediate_dim: int = 1536
    n_layers: int = 8
    istft_mult: int = 4  # iSTFT n_fft = istft_mult * hop_length
    # causal=True: every convolution is left-padded, so output frame t uses
    # input frames <= t only. The remaining lookahead is the iSTFT overlap-add
    # (center=True, n_fft = 4 hops): a sample needs the frames whose windows
    # cover it, i.e. up to 2 frames (40 ms at 50 Hz) ahead. Default False
    # (centered convs, ~27 frames of lookahead) = all earlier checkpoints.
    causal: bool = False
    # With causal=True: total lookahead of `lookahead` input frames (20 ms each
    # at 50 Hz), given to the FIRST convolutions only (embed, then block 0, ...,
    # at most kernel-1 = 6 frames each); every later layer stays causal, so the
    # output at frame t depends on input frames <= t + lookahead (+2 for iSTFT).
    lookahead: int = 0
    # Discriminators.
    mpd_periods: tuple = (2, 3, 5, 7, 11)
    mpd_channels: tuple = (32, 128, 512, 1024, 1024)
    mrd_resolutions: tuple = ((1024, 256, 1024), (2048, 512, 2048), (512, 128, 512))
    mrd_channels: int = 32
    # Losses (Vocos weights).
    mel_coef: float = 45.0
    mrd_coef: float = 0.1
    # Extra mel resolution for the L1 loss (n_fft, hop, win, n_mels); the
    # first resolution is always the input mel space itself.
    extra_mel_loss: tuple = ((1024, 240, 1024, 100),)
    # Training.
    segment_frames: int = 40  # 40 x 480 = 19,200 samples = 0.8 s at 24 kHz
    batch_size: int = 16      # per GPU
    lr: float = 5e-4
    betas: tuple = (0.8, 0.9)
    warmup_steps: int = 1_000
    max_steps: int = 500_000
    gain_db: tuple = (-6.0, -1.0)  # random peak level per utterance (Vocos)
    seed: int = 0
    log_every: int = 100
    ckpt_every: int = 10_000
    keep_every: int = 50_000
    val_every: int = 10_000
    val_utts: int = 64

    @property
    def n_fft(self) -> int:
        return self.istft_mult * self.audio.hop_length

    @property
    def segment_samples(self) -> int:
        return self.segment_frames * self.audio.hop_length

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VocoderConfig":
        d = dict(d)
        audio = AudioConfig(**d.pop("audio", {}))
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"Unknown vocoder config keys {sorted(unknown)}")
        conv = {k: (tuple(tuple(x) if isinstance(x, list) else x for x in v)
                    if isinstance(v, list) else v) for k, v in d.items()}
        return cls(audio=audio, **conv)


def tiny_vocoder_config() -> VocoderConfig:
    """CPU-sized config for tests."""
    return VocoderConfig(
        audio=AudioConfig(sample_rate=16_000, n_mels=40, n_fft=512, hop_length=320,
                          win_length=512, fmax=8_000.0),
        dim=32, intermediate_dim=64, n_layers=2,
        mpd_periods=(2, 3), mpd_channels=(4, 8, 16, 16, 16),
        mrd_resolutions=((256, 64, 256), (512, 128, 512)), mrd_channels=4,
        extra_mel_loss=((256, 80, 256, 20),),
        segment_frames=8, batch_size=2, warmup_steps=2, max_steps=4,
        log_every=1, ckpt_every=2, keep_every=0, val_every=2, val_utts=2)


def _init(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.zeros_(m.bias)


def _pad(k: int, causal: bool, ahead: int = 0) -> tuple[int, int]:
    if not causal:
        return ((k - 1) // 2, k - 1 - (k - 1) // 2)
    return (k - 1 - ahead, ahead)


def _lookahead_split(total: int, n_convs: int, k: int = 7) -> list[int]:
    """Right padding per conv (embed first) summing to `total`, each <= k - 1."""
    out = []
    for _ in range(n_convs):
        a = min(total, k - 1)
        out.append(a)
        total -= a
    if total:
        raise ValueError("lookahead larger than the generator's convolutions can provide")
    return out


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, inter: int, layer_scale: float, causal: bool = False,
                 ahead: int = 0):
        super().__init__()
        self.pad = _pad(7, causal, ahead)
        self.dw = nn.Conv1d(dim, dim, 7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, inter)
        self.pw2 = nn.Linear(inter, dim)
        self.gamma = nn.Parameter(torch.full((dim,), layer_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        y = self.dw(F.pad(x, self.pad)).transpose(1, 2)
        y = self.pw2(F.gelu(self.pw1(self.norm(y))))
        return x + (self.gamma * y).transpose(1, 2)


class Generator(nn.Module):
    """Raw log-mel (B, T, n_mels) -> waveform (B, (T - 1) * hop).

    With center=True framing on both sides, N = k * hop samples give
    T = k + 1 mel frames and the iSTFT returns exactly k * hop samples."""

    def __init__(self, cfg: VocoderConfig, in_dim: int | None = None):
        """in_dim: input feature size; None = a mel vocoder (n_mels input,
        normalized with the mel statistics). Other inputs (e.g. FaCT token
        embeddings, stage D) are used as given."""
        super().__init__()
        self.cfg = cfg
        a = cfg.audio
        self.mel_input = in_dim is None
        if cfg.lookahead and not cfg.causal:
            raise ValueError("vocoder lookahead needs causal=True (centered convs already look ahead)")
        ahead = _lookahead_split(cfg.lookahead, cfg.n_layers + 1) if cfg.causal else [0] * (cfg.n_layers + 1)
        self.embed_pad = _pad(7, cfg.causal, ahead[0])
        self.embed = nn.Conv1d(a.n_mels if in_dim is None else in_dim, cfg.dim, 7)
        self.norm = nn.LayerNorm(cfg.dim, eps=1e-6)
        self.blocks = nn.ModuleList(ConvNeXtBlock(cfg.dim, cfg.intermediate_dim, 1 / cfg.n_layers,
                                                  causal=cfg.causal, ahead=ahead[i + 1])
                                    for i in range(cfg.n_layers))
        self.final_norm = nn.LayerNorm(cfg.dim, eps=1e-6)
        self.head = nn.Linear(cfg.dim, cfg.n_fft + 2)
        self.register_buffer("window", torch.hann_window(cfg.n_fft), persistent=False)
        self.apply(_init)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        a = self.cfg.audio
        x = (mel - a.mel_mean) / a.mel_std if self.mel_input else mel
        x = self.embed(F.pad(x.transpose(1, 2), self.embed_pad))
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        x = self.head(self.final_norm(x.transpose(1, 2))).transpose(1, 2)  # (B, n_fft+2, T)
        log_mag, phase = x.float().chunk(2, dim=1)
        mag = torch.exp(log_mag).clamp(max=1e2)  # Vocos clip
        spec = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
        return torch.istft(spec, self.cfg.n_fft, hop_length=a.hop_length,
                           win_length=self.cfg.n_fft, window=self.window, center=True)


# ------------------------------------------------------------ discriminators

class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, ch: tuple):
        super().__init__()
        self.period = period
        chans = (1, *ch)
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv2d(chans[i], chans[i + 1], (5, 1),
                                  (3, 1) if i < len(ch) - 1 else (1, 1), padding=(2, 0)))
            for i in range(len(ch)))
        self.post = weight_norm(nn.Conv2d(ch[-1], 1, (3, 1), padding=(1, 0)))

    def forward(self, x: torch.Tensor):
        b, n = x.shape
        if n % self.period:
            x = F.pad(x, (0, self.period - n % self.period), mode="reflect")
        x = x.view(b, 1, -1, self.period)
        fmap = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x.flatten(1), fmap


class ResolutionDiscriminator(nn.Module):
    def __init__(self, n_fft: int, hop: int, win: int, ch: int):
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        self.register_buffer("window", torch.hann_window(win), persistent=False)
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, ch, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(ch, ch, (3, 3), padding=(1, 1))),
        ])
        self.post = weight_norm(nn.Conv2d(ch, 1, (3, 3), padding=(1, 1)))

    def forward(self, x: torch.Tensor):
        mag = torch.stft(x.float(), self.n_fft, hop_length=self.hop, win_length=self.win,
                         window=self.window, center=True, return_complex=True).abs()
        x = mag.transpose(1, 2).unsqueeze(1)  # (B, 1, T, F)
        fmap = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x.flatten(1), fmap


class MultiDiscriminator(nn.Module):
    """Runs every sub-discriminator; returns (scores, fmaps) lists."""

    def __init__(self, subs: list[nn.Module]):
        super().__init__()
        self.subs = nn.ModuleList(subs)

    def forward(self, x: torch.Tensor):
        scores, fmaps = [], []
        for d in self.subs:
            s, f = d(x)
            scores.append(s)
            fmaps.append(f)
        return scores, fmaps


def build_discriminators(cfg: VocoderConfig) -> tuple[MultiDiscriminator, MultiDiscriminator]:
    mpd = MultiDiscriminator([PeriodDiscriminator(p, cfg.mpd_channels) for p in cfg.mpd_periods])
    mrd = MultiDiscriminator([ResolutionDiscriminator(n, h, w, cfg.mrd_channels)
                              for n, h, w in cfg.mrd_resolutions])
    return mpd, mrd


# -------------------------------------------------------------------- losses
# Hinge GAN losses, averaged over sub-discriminators (as in Vocos). Each
# discriminator sees real and fake in ONE forward (batch-concatenated), so a
# score list holds [real; fake] along the batch dimension.

def split(xs: list[torch.Tensor], b: int) -> tuple[list, list]:
    return [x[:b] for x in xs], [x[b:] for x in xs]


def discriminator_loss(real: list[torch.Tensor], fake: list[torch.Tensor]) -> torch.Tensor:
    return sum(F.relu(1 - r).mean() + F.relu(1 + f).mean() for r, f in zip(real, fake)) / len(real)


def generator_adv_loss(fake: list[torch.Tensor]) -> torch.Tensor:
    return sum(F.relu(1 - f).mean() for f in fake) / len(fake)


def feature_matching_loss(fmaps: list[list[torch.Tensor]], b: int) -> torch.Tensor:
    total = 0.0
    for layers in fmaps:
        total = total + sum((l[:b].detach() - l[b:]).abs().mean() for l in layers) / len(layers)
    return total / len(fmaps)


class MelLoss(nn.Module):
    """Mean L1 of log-mels over resolutions: the input mel space first (the
    space FaCT is scored in), then cfg.extra_mel_loss."""

    def __init__(self, cfg: VocoderConfig):
        super().__init__()
        a = cfg.audio
        mels = [LogMelSpectrogram(a)]
        for n_fft, hop, win, n_mels in cfg.extra_mel_loss:
            mels.append(LogMelSpectrogram(AudioConfig(
                sample_rate=a.sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop,
                win_length=win, fmin=a.fmin, fmax=a.fmax)))
        self.mels = nn.ModuleList(mels)

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        return sum((m(y) - m(y_hat)).abs().mean() for m in self.mels) / len(self.mels)
