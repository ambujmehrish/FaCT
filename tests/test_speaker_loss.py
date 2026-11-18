"""Speaker-consistency loss for the stage-D decoder (phase 2)."""

import sys

import pytest
import torch

from fact.config import tiny_config
from fact.model import FaCT
from fact.vocoder import speaker_loss as sl


def _toy_embedder():
    torch.manual_seed(0)
    conv = torch.nn.Conv1d(1, 8, 64, stride=32)
    for p in conv.parameters():
        p.requires_grad_(False)
    calls = []

    def embed(wav):
        calls.append(wav.shape)
        return conv(wav[:, None]).mean(-1)
    return embed, calls


def test_loss_zero_for_identical_audio_and_gradient_only_into_output():
    embed, _ = _toy_embedder()
    loss = sl.SpeakerConsistencyLoss(embed, 16_000)
    y = torch.randn(3, 4000)
    assert float(loss(y, y.clone())) == pytest.approx(0.0, abs=1e-6)
    y_ref = torch.randn(3, 4000, requires_grad=True)
    y_hat = torch.randn(3, 4000, requires_grad=True)
    out = loss(y_ref, y_hat)
    assert 0.0 < float(out) <= 2.0
    out.backward()
    assert y_ref.grad is None  # the real segment is only a target
    assert y_hat.grad is not None and y_hat.grad.abs().sum() > 0


def test_loss_resamples_to_16k():
    embed, calls = _toy_embedder()
    loss = sl.SpeakerConsistencyLoss(embed, 24_000)
    loss(torch.randn(2, 24_000), torch.randn(2, 24_000))
    assert calls == [(2, 16_000), (2, 16_000)]


def test_refuses_evaluation_model_and_unknown_names():
    from fact.eval.quality import SV_MODEL
    for name in (SV_MODEL, "wavlm"):
        with pytest.raises(ValueError, match="evaluation"):
            sl.build_speaker_embedder(name, "cpu")
    with pytest.raises(ValueError, match="unknown"):
        sl.build_speaker_embedder("nobody", "cpu")
    assert SV_MODEL not in sl.BACKENDS.values()


def _setup(tmp_path):
    from tests.test_stage_d import _corpus
    cfg = tiny_config()
    torch.manual_seed(0)
    fck = tmp_path / "fact.pt"
    torch.save({"model": FaCT(cfg).state_dict(), "step": 1, "config": cfg.to_dict()}, fck)
    return fck, _corpus(tmp_path)


def test_stage_d_trains_with_speaker_loss_and_records_it(tmp_path, monkeypatch, capsys):
    from fact.train import stage_d_wave_decoder as sd
    fck, m = _setup(tmp_path)
    embed, calls = _toy_embedder()
    monkeypatch.setattr(sl, "build_speaker_embedder", lambda name, device: embed)
    ck = tmp_path / "ck"
    base = ["x", "--tiny", "--fact-ckpt", str(fck), "--manifest", str(m), "--dev-manifest", str(m),
            "--ckpt-dir", str(ck), "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", base + ["--steps", "2", "--spk-loss-coef", "0.5"])
    sd.main()
    saved = torch.load(ck / "step_0000002.pt", weights_only=True)
    assert saved["run"]["spk_loss"] == {"coef": 0.5, "model": "ecapa"}
    assert len(calls) == 4  # target + output, 2 steps
    # Resuming with a different weight is refused.
    monkeypatch.setattr(sys, "argv", base + ["--steps", "3", "--spk-loss-coef", "1.0",
                                             "--resume", str(ck / "step_0000002.pt")])
    with pytest.raises(SystemExit, match="spk_loss"):
        sd.main()


def test_stage_d_without_speaker_loss_records_nothing(tmp_path, monkeypatch):
    from fact.train import stage_d_wave_decoder as sd
    fck, m = _setup(tmp_path)

    def boom(*a, **k):
        raise AssertionError("embedder built although the loss is off")
    monkeypatch.setattr(sl, "build_speaker_embedder", boom)
    ck = tmp_path / "ck"
    monkeypatch.setattr(sys, "argv", ["x", "--tiny", "--fact-ckpt", str(fck), "--manifest", str(m),
                                      "--dev-manifest", str(m), "--ckpt-dir", str(ck),
                                      "--device", "cpu", "--steps", "2"])
    sd.main()
    assert "spk_loss" not in torch.load(ck / "step_0000002.pt", weights_only=True)["run"]


def test_negative_coefficient_refused_before_loading(monkeypatch):
    from fact.train import stage_d_wave_decoder as sd
    monkeypatch.setattr(sys, "argv", ["x", "--fact-ckpt", "missing.pt", "--manifest", "m",
                                      "--dev-manifest", "m", "--ckpt-dir", "c", "--device", "cpu",
                                      "--spk-loss-coef", "-1"])
    with pytest.raises(SystemExit, match=">= 0"):
        sd.main()
