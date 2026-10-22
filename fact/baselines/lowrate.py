"""Dedicated low-bitrate speech codecs: the competitors at FaCT's bitrate.

    focalcodec:config=lucadellalib/focalcodec_50hz            0.65 kbps, 16 kHz
    focalcodec:config=lucadellalib/focalcodec_25hz            0.33 kbps
    focalcodec:config=lucadellalib/focalcodec_12_5hz          0.16 kbps
    focalcodec:config=lucadellalib/focalcodec_50hz_4k_causal  0.60 kbps, causal (16 in / 24 out)
    wavtokenizer:variant=large_speech_75                      0.90 kbps, 24 kHz
    wavtokenizer:variant=large_unify_40                       0.48 kbps, 24 kHz
    bigcodec                                                  1.04 kbps, 16 kHz

Each runs the authors' own inference code from a pinned git commit (the
repos are not pip packages, or would pin old torch), cloned on a login node
by scripts/fetch_lowrate_codecs.sh into $FACT_CACHE/codecs/<name>; weights
come from the Hugging Face cache (offline on compute nodes).

Frame rate and codebook size are read from the loaded model (frame rate by
encoding a 10 s probe), never hardcoded, so the bitrate in a table is the
bitrate of the checkpoint that produced the numbers.

Pickled checkpoints (BigCodec, WavTokenizer) are loaded with torch's
weights-only unpickler (tensors and plain containers only), so no code from
the file is executed; FocalCodec weights are safetensors.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch

from .base import register

# Pinned commits (scripts/fetch_lowrate_codecs.sh clones exactly these).
REPOS = {
    "focalcodec": ("https://github.com/lucadellalib/focalcodec", "912b7f2"),
    "WavTokenizer": ("https://github.com/jishengpeng/WavTokenizer", "5cf440d"),
    "BigCodec": ("https://github.com/Aria-K-Alethia/BigCodec", "09845ab"),
    "SpeechMOS": ("https://github.com/tarepan/SpeechMOS", "ed25eac"),  # UTMOS (fact/eval/quality.py)
}


def codec_source(name: str) -> Path:
    root = os.environ.get("FACT_CODEC_SRC") or (
        str(Path(os.environ["FACT_CACHE"]) / "codecs") if os.environ.get("FACT_CACHE") else None)
    if root is None or not (Path(root) / name).is_dir():
        raise ImportError(f"{name} source not found under {root}; run "
                          "scripts/fetch_lowrate_codecs.sh on a login node")
    path = Path(root) / name
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def load_checkpoint(path: str) -> dict:
    """torch.load with the weights-only unpickler: tensors and plain
    containers only, so no code stored in the file can run. The BigCodec and
    WavTokenizer checkpoints load this way as published; anything else fails
    loudly instead of falling back to full unpickling."""
    return torch.load(path, map_location="cpu", weights_only=True)


def measured_frame_rate(encode, sample_rate: int, device: str, seconds: float = 10.0) -> float:
    """Tokens per second of `encode` on a probe signal, to 0.25 Hz."""
    g = torch.Generator().manual_seed(0)
    probe = (0.05 * torch.randn(1, int(seconds * sample_rate), generator=g)).to(device)
    with torch.no_grad():
        n = encode(probe).shape[-1]
    return round(4 * n / seconds) / 4


class FocalCodecBaseline:
    def __init__(self, config: str = "lucadellalib/focalcodec_50hz", device: str = "cuda"):
        codec_source("focalcodec")
        # FocalCodec falls back to unpickling model.pt when safetensors is
        # missing; require it so only the safetensors weights are ever read.
        import safetensors.torch  # noqa: F401
        from focalcodec import FocalCodec
        self.model = FocalCodec.from_config(config, pretrained=True).to(device).eval()
        self.name = f"focalcodec:{config}"
        self.device = device
        self.sample_rate = self.model.sample_rate_input
        self.out_rate = self.model.sample_rate_output
        self.frame_rate_hz = measured_frame_rate(self.model.sig_to_toks, self.sample_rate, device)
        cb = int(self.model.codebook.shape[0])
        self.vocab_sizes = [cb]
        self.discrete_bps = math.log2(cb) * self.frame_rate_hz

    @torch.no_grad()
    def tokenize(self, wav):
        return [self.model.sig_to_toks(wav.to(self.device))]

    @torch.no_grad()
    def reconstruct(self, wav):
        wav = wav.to(self.device)
        _, _, rec = self.model(wav)
        if self.out_rate != self.sample_rate:  # causal variants decode at 24 kHz
            from ..eval.metrics_audio import resample
            rec = resample(rec.float().cpu(), self.out_rate, self.sample_rate).to(self.device)
        return rec[..., : wav.shape[-1]]


WAVTOKENIZER = {  # variant -> (HF repo, checkpoint file, config in the repo)
    "large_speech_75": ("novateur/WavTokenizer-large-speech-75token",
                        "wavtokenizer_large_speech_320_v2.ckpt",
                        "wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"),
    "large_unify_40": ("novateur/WavTokenizer-large-unify-40token",
                       "wavtokenizer_large_unify_600_24k.ckpt",
                       "wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"),
}


class WavTokenizerBaseline:
    def __init__(self, variant: str = "large_speech_75", device: str = "cuda"):
        if variant not in WAVTOKENIZER:
            raise ValueError(f"variant must be one of {sorted(WAVTOKENIZER)}")
        src = codec_source("WavTokenizer")
        from huggingface_hub import hf_hub_download
        from decoder.pretrained import WavTokenizer
        repo, fname, cfg_name = WAVTOKENIZER[variant]
        model = WavTokenizer.from_hparams0802(str(src / "configs" / cfg_name))
        sd = load_checkpoint(hf_hub_download(repo, fname))["state_dict"]
        # Same key selection as the authors' from_pretrained0802.
        keep = {k: v for k, v in sd.items()
                if k.startswith(("backbone.", "head.", "feature_extractor."))}
        model.load_state_dict(keep)
        self.model = model.to(device).eval()
        self.name = f"wavtokenizer:{variant}"
        self.device = device
        self.sample_rate = 24_000
        self.bw = torch.tensor([0], device=device)
        self.frame_rate_hz = measured_frame_rate(lambda x: self._codes(x), self.sample_rate, device)
        cb = int(self.model.feature_extractor.encodec.quantizer.bins)
        self.vocab_sizes = [cb]
        self.discrete_bps = math.log2(cb) * self.frame_rate_hz

    def _codes(self, wav):
        return self.model.encode_infer(wav, bandwidth_id=self.bw)[1]

    @torch.no_grad()
    def tokenize(self, wav):
        return [self._codes(wav.to(self.device))[0]]

    @torch.no_grad()
    def reconstruct(self, wav):
        wav = wav.to(self.device)
        feats, _ = self.model.encode_infer(wav, bandwidth_id=self.bw)
        return self.model.decode(feats, bandwidth_id=self.bw)[..., : wav.shape[-1]]


class BigCodecBaseline:
    def __init__(self, device: str = "cuda"):
        codec_source("BigCodec")
        from huggingface_hub import hf_hub_download
        from vq.codec_decoder import CodecDecoder
        from vq.codec_encoder import CodecEncoder
        ck = load_checkpoint(hf_hub_download("Alethia/BigCodec", "bigcodec.pt"))
        self.enc, self.dec = CodecEncoder(), CodecDecoder()
        self.enc.load_state_dict(ck["CodecEnc"])
        self.dec.load_state_dict(ck["generator"])
        self.enc, self.dec = self.enc.to(device).eval(), self.dec.to(device).eval()
        self.name = "bigcodec"
        self.device = device
        self.sample_rate = 16_000
        self.hop = int(self.enc.hop_length)
        self.frame_rate_hz = measured_frame_rate(self._codes, self.sample_rate, device)
        layers = self.dec.quantizer.layers  # ResidualVQ with one factorized VQ
        if len(layers) != 1:
            raise ValueError(f"expected a single-codebook BigCodec, got {len(layers)} quantizers")
        cb = int(layers[0].codebook_size)
        self.vocab_sizes = [cb]
        self.discrete_bps = math.log2(cb) * self.frame_rate_hz

    def _pad(self, wav):
        # The authors' inference pads to a multiple of the hop (always adding
        # at least one frame); the output is cut back to the input length.
        return torch.nn.functional.pad(wav, (0, self.hop - wav.shape[-1] % self.hop))

    def _codes(self, wav):
        _, codes, _ = self.dec(self.enc(self._pad(wav).unsqueeze(1)), vq=True)
        return codes

    @torch.no_grad()
    def tokenize(self, wav):
        return [self._codes(wav.to(self.device)).reshape(wav.shape[0], -1)]

    @torch.no_grad()
    def reconstruct(self, wav):
        wav = wav.to(self.device)
        post, _, _ = self.dec(self.enc(self._pad(wav).unsqueeze(1)), vq=True)
        return self.dec(post, vq=False).squeeze(1)[..., : wav.shape[-1]]


@register("focalcodec")
def _focal(device: str = "cuda", config: str = "lucadellalib/focalcodec_50hz"):
    return FocalCodecBaseline(config=config, device=device)


@register("wavtokenizer")
def _wavtok(device: str = "cuda", variant: str = "large_speech_75"):
    return WavTokenizerBaseline(variant=variant, device=device)


@register("bigcodec")
def _bigcodec(device: str = "cuda"):
    return BigCodecBaseline(device=device)
