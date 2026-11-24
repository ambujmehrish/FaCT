"""FACodec (NaturalSpeech 3): the factorized-codec reference baseline.

    facodec                    all six code streams (prosody 1, content 2, detail 3)
    facodec:view=no_residual   prosody + content codes only (no acoustic-detail codes)

FACodec factorizes speech into prosody, content and acoustic-detail code
streams plus a global timbre vector (GRL supervision), like FaCT, but it is
non-causal and runs at much higher bitrate: 80 Hz, 6 codebooks of 2^10 at
the published checkpoint (read from the model, not hardcoded). The timbre
vector is computed from the utterance itself (no enrolment), and it is not
counted in the bitrate, as FaCT's speaker vector is not.

Code: Amphion's `models/codec/ns3_codec` at a pinned commit, cloned by
scripts/fetch_lowrate_codecs.sh into $FACT_CACHE/codecs/Amphion and imported
by file path (the rest of Amphion is never imported). Weights:
HF `amphion/naturalspeech3_facodec` (ns3_facodec_{encoder,decoder}.bin),
loaded with torch's weights-only unpickler.

Beyond reconstruct()/tokenize(), `encode()` / `decode()` expose the streams
so the factorization and voice-conversion evaluations can swap them exactly
as they swap FaCT's (scripts/eval_facodec.py).
"""

from __future__ import annotations

import importlib.util
import math
import sys

import torch

from .base import register

HF_REPO = "amphion/naturalspeech3_facodec"
VIEWS = ("full", "no_residual")
# The published checkpoint's constructor arguments (Amphion ns3_codec README).
ENCODER_KW = dict(ngf=32, up_ratios=[2, 4, 5, 5], out_channels=256)
DECODER_KW = dict(in_channels=256, upsample_initial_channel=1024, ngf=32, up_ratios=[5, 5, 4, 2],
                  vq_num_q_c=2, vq_num_q_p=1, vq_num_q_r=3, vq_dim=256, codebook_dim=8,
                  codebook_size_prosody=10, codebook_size_content=10, codebook_size_residual=10,
                  use_gr_x_timbre=True, use_gr_residual_f0=True, use_gr_residual_phone=True)


def _ns3_codec():
    """Import Amphion's ns3_codec package by path, nothing else from Amphion."""
    name = "fact_ext_ns3_codec"
    if name in sys.modules:
        return sys.modules[name]
    from .lowrate import codec_source  # here, not at import: lowrate imports the registry
    root = codec_source("Amphion") / "models" / "codec" / "ns3_codec"
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py",
                                                  submodule_search_locations=[str(root)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class FACodecBaseline:
    name = "facodec"
    sample_rate = 16_000

    def __init__(self, view: str = "full", device: str = "cuda"):
        if view not in VIEWS:
            raise ValueError(f"facodec view must be one of {VIEWS}")
        from huggingface_hub import hf_hub_download

        from .lowrate import load_checkpoint, measured_frame_rate
        ns3 = _ns3_codec()
        self.encoder = ns3.FACodecEncoder(**ENCODER_KW)
        self.decoder = ns3.FACodecDecoder(**DECODER_KW)
        self.encoder.load_state_dict(load_checkpoint(hf_hub_download(HF_REPO, "ns3_facodec_encoder.bin")))
        self.decoder.load_state_dict(load_checkpoint(hf_hub_download(HF_REPO, "ns3_facodec_decoder.bin")))
        self.encoder = self.encoder.to(device).eval()
        self.decoder = self.decoder.to(device).eval()
        self.view, self.device = view, device
        self.name = "facodec" if view == "full" else f"facodec_{view}"
        d = self.decoder
        self.n_streams = {"prosody": d.vq_num_q_p, "content": d.vq_num_q_c, "residual": d.vq_num_q_r}
        self.frame_rate_hz = measured_frame_rate(lambda w: self.encode(w)["ids"], self.sample_rate, device)
        sizes = self._codebook_sizes()
        keep = sizes if view == "full" else sizes[: d.vq_num_q_p + d.vq_num_q_c]
        self.vocab_sizes = keep
        self.discrete_bps = sum(math.log2(v) for v in keep) * self.frame_rate_hz

    def _codebook_sizes(self) -> list[int]:
        sizes = []
        for q in self.decoder.quantizer:  # prosody, content, residual groups (residual VQs)
            for layer in q.layers:
                sizes.append(int(layer.codebook_size))
        return sizes

    def _wav(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, N) or (N,) -> (B, 1, N') with N' a whole number of frames."""
        if wav.dim() == 1:
            wav = wav[None]
        hop = int(self.encoder.hop_length)
        n = wav.shape[-1] // hop * hop
        return wav[..., :n].to(self.device).float().unsqueeze(1)

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> dict:
        """{'ids': (n_q, B, T) codes in the order prosody, content, residual,
        'spk': (B, D) timbre vector of this utterance}."""
        enc = self.encoder(self._wav(wav))
        _, ids, _, _, spk = self.decoder(enc, eval_vq=False, vq=True)
        return {"ids": ids, "spk": spk}

    @torch.no_grad()
    def decode(self, ids: torch.Tensor, spk: torch.Tensor, use_residual: bool | None = None) -> torch.Tensor:
        """Codes (n_q, B, T) + timbre vector -> waveform (B, N) at 16 kHz."""
        use_residual = (self.view == "full") if use_residual is None else use_residual
        emb = self.decoder.vq2emb(ids, use_residual_code=use_residual)
        return self.decoder.inference(emb, spk)[:, 0]

    def stream_slices(self) -> dict[str, slice]:
        p, c = self.n_streams["prosody"], self.n_streams["content"]
        return {"prosody": slice(0, p), "content": slice(p, p + c), "residual": slice(p + c, None)}

    @torch.no_grad()
    def tokenize(self, wav):
        ids = self.encode(wav)["ids"]
        n = len(self.vocab_sizes)
        return [ids[i] for i in range(n)]

    @torch.no_grad()
    def reconstruct(self, wav):
        out = self.encode(wav)
        rec = self.decode(out["ids"], out["spk"])
        n = wav.shape[-1]
        return rec[..., :n] if rec.shape[-1] >= n else torch.nn.functional.pad(rec, (0, n - rec.shape[-1]))


@register("facodec")
def _build(device: str = "cuda", view: str = "full") -> FACodecBaseline:
    return FACodecBaseline(view=view, device=device)
