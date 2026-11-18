"""Speaker-consistency loss for waveform decoders (phase 2 of the decoder
retraining, docs/ICML_TODO.md item 0).

    loss = mean over the batch of 1 - cos(e(y_hat), e(y))

e is a FROZEN speaker-verification model; e(y) of the real segment is the
target (no gradient), the gradient flows through e into the generated audio
only. The model must differ from the one that scores speaker similarity in
evaluation (fact/eval/quality.py SV_MODEL, WavLM-base-plus-sv), otherwise
the decoder would be trained on the test metric itself.

Backends (weights fetched on a login node with
`python scripts/download_checkpoints.py --baselines ecapa`; jobs run offline):
  ecapa          speechbrain/spkrec-ecapa-voxceleb: ECAPA-TDNN on fbanks, a
                 different architecture and input from the evaluation model.
                 Needs the `speechbrain` package.
  unispeech_sat  microsoft/unispeech-sat-base-plus-sv (transformers). Same
                 x-vector recipe as the evaluation model on another
                 backbone, so less independent; only for when speechbrain
                 cannot be installed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

SV_SAMPLE_RATE = 16_000
BACKENDS = {"ecapa": "speechbrain/spkrec-ecapa-voxceleb",
            "unispeech_sat": "microsoft/unispeech-sat-base-plus-sv"}

Embedder = Callable[[torch.Tensor], torch.Tensor]  # (B, N) 16 kHz -> (B, D)


def _local_snapshot(repo: str) -> str:
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(repo, local_files_only=True)
    except Exception as e:  # noqa: BLE001 - re-raised with the fix
        raise FileNotFoundError(f"{repo} is not in the HF cache ({os.environ.get('HF_HOME')}); "
                                "run scripts/download_checkpoints.py --baselines "
                                f"{[k for k, v in BACKENDS.items() if v == repo][0]} "
                                "on a login node") from e


def _freeze(m: torch.nn.Module) -> torch.nn.Module:
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _ecapa(device: str) -> Embedder:
    """Built from the cached hyperparams.yaml and checkpoint, read-only:
    EncoderClassifier.from_hparams would write symlinks into a shared save
    dir from every rank at once. Only the parts encode_batch uses: fbank
    features, sentence mean normalisation, ECAPA-TDNN."""
    try:
        from hyperpyyaml import load_hyperpyyaml
    except ImportError as e:
        raise ImportError("spk loss 'ecapa' needs the speechbrain package (see "
                          "docs/ICML_TODO.md item 0 for the install command)") from e
    src = Path(_local_snapshot(BACKENDS["ecapa"]))
    with open(src / "hyperparams.yaml", encoding="utf-8") as fh:
        hp = load_hyperpyyaml(fh, overrides={"pretrained_path": str(src)})
    feats, norm, emb = hp["compute_features"], hp["mean_var_norm"], hp["embedding_model"]
    emb.load_state_dict(torch.load(src / "embedding_model.ckpt", map_location="cpu", weights_only=True))
    feats, norm, emb = (_freeze(m.to(device)) for m in (feats, norm, emb))

    def embed(wav: torch.Tensor) -> torch.Tensor:
        lens = torch.ones(wav.shape[0], device=wav.device)
        return emb(norm(feats(wav), lens), lens).squeeze(1)
    return embed


def _unispeech_sat(device: str) -> Embedder:
    from transformers import UniSpeechSatForXVector
    model = _freeze(UniSpeechSatForXVector.from_pretrained(_local_snapshot(BACKENDS["unispeech_sat"])).to(device))

    def embed(wav: torch.Tensor) -> torch.Tensor:
        # The feature extractor's zero-mean unit-variance normalisation, in torch.
        x = (wav - wav.mean(-1, keepdim=True)) / torch.sqrt(wav.var(-1, keepdim=True, unbiased=False) + 1e-7)
        return model(input_values=x).embeddings
    return embed


def build_speaker_embedder(name: str, device: str) -> Embedder:
    from ..eval.quality import SV_MODEL
    if name in (SV_MODEL, "wavlm", "wavlm_sv") or BACKENDS.get(name) == SV_MODEL:
        raise ValueError(f"{name}: that is the evaluation SV model; train with another one")
    if name == "ecapa":
        return _ecapa(device)
    if name == "unispeech_sat":
        return _unispeech_sat(device)
    raise ValueError(f"unknown speaker-loss model {name!r}; use one of {sorted(BACKENDS)}")


class SpeakerConsistencyLoss(torch.nn.Module):
    def __init__(self, embed: Embedder, sample_rate: int):
        super().__init__()
        self.embed, self.sample_rate = embed, sample_rate

    def _to_sv_rate(self, wav: torch.Tensor) -> torch.Tensor:
        if self.sample_rate == SV_SAMPLE_RATE:
            return wav
        import torchaudio
        return torchaudio.functional.resample(wav, self.sample_rate, SV_SAMPLE_RATE)

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """y, y_hat: (B, N) at sample_rate. Gradient only through y_hat."""
        with torch.no_grad():
            target = self.embed(self._to_sv_rate(y.float()))
        pred = self.embed(self._to_sv_rate(y_hat.float()))
        return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
