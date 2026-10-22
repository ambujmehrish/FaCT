"""Stage-C vocoder: shapes, losses, training/resume, data stream, eval wrapper."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from fact.config import tiny_config
from fact.model import FaCT
from fact.vocoder.model import (Generator, MelLoss, VocoderConfig, build_discriminators,
                                discriminator_loss, feature_matching_loss,
                                generator_adv_loss, split, tiny_vocoder_config)


def test_generator_length_matches_input_audio():
    for vc in (tiny_vocoder_config(), VocoderConfig()):
        gen = Generator(vc).eval()
        from fact.modules.mel import LogMelSpectrogram
        n = 6 * vc.audio.hop_length
        mel = LogMelSpectrogram(vc.audio)(torch.randn(2, n))
        assert mel.shape[1] == 7  # center=True: N // hop + 1 frames
        with torch.no_grad():
            assert gen(mel).shape == (2, n)
    assert VocoderConfig().n_fft == 1920  # 4 x hop 480


def test_losses_and_discriminators():
    vc = tiny_vocoder_config()
    mpd, mrd = build_discriminators(vc)
    y, y_hat = torch.randn(2, vc.segment_samples), torch.randn(2, vc.segment_samples)
    for d in (mpd, mrd):
        scores, fmaps = d(torch.cat([y, y_hat]))
        assert len(scores) == len(d.subs)
        real, fake = split(scores, 2)
        assert torch.isfinite(discriminator_loss(real, fake))
        assert torch.isfinite(generator_adv_loss(fake))
        assert torch.isfinite(feature_matching_loss(fmaps, 2))
    # Identical real and fake: feature matching and mel loss vanish.
    _, fmaps = mpd(torch.cat([y, y]))
    assert feature_matching_loss(fmaps, 2).item() == pytest.approx(0.0, abs=1e-7)
    assert MelLoss(vc)(y, y).item() == pytest.approx(0.0, abs=1e-7)


def test_config_round_trip():
    vc = tiny_vocoder_config()
    assert VocoderConfig.from_dict(vc.to_dict()) == vc
    with pytest.raises(KeyError):
        VocoderConfig.from_dict({**vc.to_dict(), "bogus": 1})


def _corpus(tmp_path, n=6, seconds=1.0, sr=16_000):
    import soundfile as sf
    rng = np.random.default_rng(0)
    lines = []
    for k in range(n):
        t = np.arange(int(seconds * sr)) / sr
        x = 0.3 * np.sin(2 * np.pi * (120 + 40 * k) * t) + 0.01 * rng.standard_normal(len(t))
        p = tmp_path / f"u{k}.wav"
        sf.write(p, x.astype("float32"), sr)
        lines.append(f"{p}\ttext {k}\n")
    m = tmp_path / "m.tsv"
    m.write_text("".join(lines), encoding="utf-8")
    return m


def test_random_segments_length_and_level(tmp_path):
    from fact.vocoder.data import RandomSegments
    m = _corpus(tmp_path)
    ds = RandomSegments(str(m), 16_000, 2_560, seed=0, gain_db=(-6.0, -1.0))
    it = iter(ds)
    for _ in range(10):
        seg = next(it)
        assert seg.shape == (2_560,)
        assert seg.abs().max() <= 10 ** (-1 / 20) + 1e-6
    # Different resume offsets draw different streams.
    a = next(iter(RandomSegments(str(m), 16_000, 2_560, seed=0, step_offset=0)))
    b = next(iter(RandomSegments(str(m), 16_000, 2_560, seed=0, step_offset=7)))
    assert not torch.equal(a, b)


def _train(monkeypatch, args):
    from fact.train import stage_c_vocoder
    monkeypatch.setattr(sys, "argv", ["stage_c_vocoder", *args])
    stage_c_vocoder.main()


def test_train_save_resume_and_guard(tmp_path, monkeypatch):
    m = _corpus(tmp_path)
    ck = tmp_path / "ck"
    base = ["--tiny", "--manifest", str(m), "--dev-manifest", str(m),
            "--ckpt-dir", str(ck), "--device", "cpu", "--steps", "4"]
    _train(monkeypatch, base)
    assert sorted(p.name for p in ck.glob("step_*.pt")) == ["step_0000002.pt", "step_0000004.pt"]
    s4 = torch.load(ck / "step_0000004.pt", weights_only=True)
    assert s4["step"] == 4 and all(torch.isfinite(v).all() for v in s4["generator"].values())
    # Resume from step 2 reaches step 4 again.
    (ck / "step_0000004.pt").unlink()
    _train(monkeypatch, base + ["--resume", str(ck / "step_0000002.pt")])
    assert (ck / "step_0000004.pt").exists()
    # A changed vocoder setting refuses to resume.
    with pytest.raises(SystemExit):
        _train(monkeypatch, base + ["--resume", str(ck / "step_0000002.pt"),
                                    "--set", "vocoder.mel_coef=10"])


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(f"scripts/{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fact_vocoded_wrapper(tmp_path, monkeypatch):
    from fact.baselines import base as registry
    m = _corpus(tmp_path)
    ck = tmp_path / "voc"
    _train(monkeypatch, ["--tiny", "--manifest", str(m), "--dev-manifest", str(m),
                         "--ckpt-dir", str(ck), "--device", "cpu", "--steps", "2"])
    voc = ck / "step_0000002.pt"
    cfg = tiny_config()  # same mel space as tiny_vocoder_config
    cfg.bottleneck.residual_capacity = 2.0
    model = FaCT(cfg)
    fck = tmp_path / "fact.pt"
    torch.save({"model": model.state_dict(), "step": 1, "config": cfg.to_dict()}, fck)
    wav = 0.1 * torch.randn(1, 16 * 320)
    tok_bps = math.log2(125) * 12.5 + math.log2(16) * 12.5
    for view in ("full", "tokens_prior", "discrete_only"):
        tok = registry.build("fact", device="cpu", ckpt=str(fck), vocoder=str(voc),
                             view=view, nfe=2)
        out = tok.reconstruct(wav)
        # 17 mel frames, cropped to 16 (whole tokens) -> 15 hops of audio.
        assert out.shape == (1, 15 * 320) and torch.isfinite(out).all()
        want = tok_bps + (2.0 / math.log(2) * 12.5 if view == "full" else 0.0)
        assert tok.discrete_bps == pytest.approx(want)
    # Same seed and utterance index -> same output.
    a = registry.build("fact", device="cpu", ckpt=str(fck), vocoder=str(voc), nfe=2).reconstruct(wav)
    b = registry.build("fact", device="cpu", ckpt=str(fck), vocoder=str(voc), nfe=2).reconstruct(wav)
    assert torch.equal(a, b)
    cs = registry.build("fact_vocoder", device="cpu", vocoder=str(voc))
    assert cs.reconstruct(wav).shape == (1, 16 * 320) and cs.discrete_bps is None
    # A FaCT checkpoint in another mel space is refused.
    cfg2 = tiny_config()
    cfg2.audio.n_mels = 32
    torch.save({"model": FaCT(cfg2).state_dict(), "step": 1, "config": cfg2.to_dict()},
               tmp_path / "f2.pt")
    with pytest.raises(ValueError):
        registry.build("fact", device="cpu", ckpt=str(tmp_path / "f2.pt"), vocoder=str(voc))


def test_spec_label_uses_label_field():
    eb = _load_script("eval_baselines")
    spec = "fact:ckpt=/a/b/step.pt:vocoder=/c/v.pt:view=full:label=fact_main_full"
    assert eb.spec_label(spec) == "fact_main_full"
    assert eb.spec_label("mimi:n_streams=2") == "mimi_n_streams2"
