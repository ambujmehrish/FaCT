"""Model-based quality metrics: UTMOS (naturalness) and speaker similarity.

  utmos    UTMOS22 strong learner (Saeki et al., 2022), via the SpeechMOS
           port (tarepan/SpeechMOS, pinned commit, MIT). Score of the
           reconstruction alone, on a 1-5 MOS scale.
  spk_sim  cosine similarity of speaker embeddings (reference vs
           reconstruction) from microsoft/wavlm-base-plus-sv
           (WavLMForXVector).

Both run on the GPU, offline: scripts/fetch_lowrate_codecs.sh clones
SpeechMOS and fetches its weights into $FACT_CACHE/codecs/SpeechMOS and the
WavLM checkpoint into the Hugging Face cache.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

UTMOS_WEIGHTS = "utmos22_strong_step7459_v1.pt"
UTMOS_URL = f"https://github.com/tarepan/SpeechMOS/releases/download/v1.0.0/{UTMOS_WEIGHTS}"
SV_MODEL = "microsoft/wavlm-base-plus-sv"


class UTMOS:
    def __init__(self, device: str = "cuda"):
        from ..baselines.lowrate import codec_source
        src = codec_source("SpeechMOS")
        from speechmos.utmos22.strong.model import UTMOS22Strong
        weights = Path(src) / UTMOS_WEIGHTS
        if not weights.exists():
            raise FileNotFoundError(f"{weights} missing; run scripts/fetch_lowrate_codecs.sh")
        self.model = UTMOS22Strong()
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, wav: torch.Tensor, sr: int) -> float:
        """wav: (N,) at sr."""
        return float(self.model(wav.float().to(self.device).unsqueeze(0), sr)[0])


class SpeakerSim:
    def __init__(self, device: str = "cuda"):
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.fe = AutoFeatureExtractor.from_pretrained(SV_MODEL)
        self.model = WavLMForXVector.from_pretrained(SV_MODEL).to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        from .metrics_audio import resample
        w = resample(wav.detach().float().cpu(), sr, 16_000).numpy()
        inp = self.fe(w, sampling_rate=16_000, return_tensors="pt")
        return self.model(**{k: v.to(self.device) for k, v in inp.items()}).embeddings[0]

    def __call__(self, ref: torch.Tensor, hyp: torch.Tensor, sr: int) -> float:
        return float(F.cosine_similarity(self.embed(ref, sr), self.embed(hyp, sr), dim=0))


def build_quality_hook(utmos: bool, sim: bool, device: str):
    """hook(ref, hyp, sr) -> {"utmos": .., "spk_sim": ..} for the enabled metrics."""
    if not (utmos or sim):
        return None
    mos = UTMOS(device) if utmos else None
    sv = SpeakerSim(device) if sim else None

    def hook(ref: torch.Tensor, hyp: torch.Tensor, sr: int) -> dict:
        out = {}
        if mos is not None:
            out["utmos"] = mos(hyp, sr)
        if sv is not None:
            out["spk_sim"] = sv(ref, hyp, sr)
        return out
    return hook
