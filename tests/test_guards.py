"""Guards against silent failure modes - the pre-flight checks for real runs."""

import json
import math

import numpy as np
import pytest
import torch

from fact.config import tiny_config
from fact.data.dataset import verify_mel_stats
from fact.data.preprocess import load_audio, preprocess
from fact.data.prosody_targets import extract_f0
from fact.modules.heads import CTCHead


# ---------------------------------------------------------------- CTC feasibility

def test_ctc_upsample_logit_length():
    cfg = tiny_config().ctc
    head = CTCHead(16, cfg)
    x = torch.randn(2, 10, 16)
    assert head.logits(x).shape == (2, 10 * cfg.upsample, cfg.vocab_size)


def test_ctc_reports_infeasible_fraction():
    cfg = tiny_config().ctc  # upsample = 2
    head = CTCHead(16, cfg)
    x = torch.randn(2, 5, 16)  # 10 CTC positions after upsampling
    targets = torch.randint(1, cfg.vocab_size, (2, 12))
    # Item 0 feasible (8 <= 10), item 1 infeasible (12 > 10).
    lengths = torch.tensor([8, 12])
    loss, infeasible = head(x, targets, lengths)
    assert torch.isfinite(loss)
    assert abs(float(infeasible) - 0.5) < 1e-6


def test_ctc_feasible_at_realistic_byte_rate():
    """12.5 Hz tokens x upsample 2 = 25 pos/s must cover ~15 bytes/s English."""
    cfg = tiny_config().ctc
    head = CTCHead(16, cfg)
    seconds = 4
    t_tok = int(12.5 * seconds)
    n_bytes = int(15 * seconds)  # realistic byte-level transcript length
    x = torch.randn(1, t_tok, 16)
    targets = torch.randint(1, cfg.vocab_size, (1, n_bytes))
    loss, infeasible = head(x, targets, torch.tensor([n_bytes]))
    assert float(infeasible) == 0.0
    assert float(loss) > 0


# ---------------------------------------------------------------- F0 explicitness

def test_extract_f0_rejects_unknown_method():
    wav = np.zeros(16_000, dtype=np.float32)
    with pytest.raises(ValueError):
        extract_f0(wav, 16_000, 320, 50, 550, method="magic")


def test_extract_f0_autocorr_is_explicit_opt_in():
    sr = 16_000
    t = np.arange(sr) / sr
    wav = 0.5 * np.sin(2 * math.pi * 200 * t).astype(np.float32)
    f0 = extract_f0(wav, sr, 320, 50, 550, method="autocorr")
    assert (f0 >= 0).all()


# ---------------------------------------------------------------- audio decoding

def test_load_audio_error_carries_causes(tmp_path):
    bad = tmp_path / "not_audio.wav"
    bad.write_bytes(b"this is not a wav file")
    with pytest.raises(RuntimeError, match="soundfile"):
        load_audio(str(bad), 16_000)


# ---------------------------------------------------------------- drop-rate guard

def test_preprocess_aborts_on_systematic_drops(tmp_path):
    manifest = tmp_path / "m.tsv"
    manifest.write_text("".join(f"/nonexistent/{i}.wav\ttext\n" for i in range(8)))
    cfg = tiny_config()
    # f0_method explicit so this test doesn't depend on the optional pyworld.
    with pytest.raises(SystemExit, match="ABORT"):
        preprocess(str(manifest), str(tmp_path / "shards"), cfg,
                   shard_size=4, workers=1, f0_method="autocorr",
                   max_drop_rate=0.05)
    assert (tmp_path / "shards" / "drops.log").exists()


def test_preprocess_missing_pyworld_message():
    """The pyworld pre-flight names the escape hatch explicitly."""
    import fact.data.preprocess as pp
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "pyworld":
            raise ImportError("no pyworld")
        return real_import(name, *a, **k)

    builtins.__import__ = fake_import
    try:
        with pytest.raises(SystemExit, match="autocorr"):
            pp.preprocess("unused.tsv", "unused", tiny_config(), f0_method="pyworld")
    finally:
        builtins.__import__ = real_import


# ---------------------------------------------------------------- stats guard

def _write_stats(d, mean, std, fingerprint="match"):
    from fact.data.preprocess import config_fingerprint
    d.mkdir(parents=True, exist_ok=True)
    stats = {"n_utterances": 10, "hours": 1.0, "mel_mean": mean, "mel_std": std}
    if fingerprint == "match":
        stats["fingerprint"] = config_fingerprint(tiny_config())
    elif fingerprint is not None:
        stats["fingerprint"] = fingerprint
    (d / "stats.json").write_text(json.dumps(stats))


def test_verify_mel_stats_rejects_mismatch(tmp_path):
    cfg = tiny_config()  # defaults mel_mean=-4.0, mel_std=3.0
    _write_stats(tmp_path, mean=-7.5, std=3.0)
    with pytest.raises(SystemExit, match="does not match"):
        verify_mel_stats(str(tmp_path), cfg)


def test_verify_mel_stats_accepts_match_and_override(tmp_path):
    cfg = tiny_config()
    _write_stats(tmp_path, mean=cfg.audio.mel_mean + 0.1, std=cfg.audio.mel_std - 0.1)
    verify_mel_stats(str(tmp_path), cfg)  # within tolerance
    _write_stats(tmp_path, mean=-9.0, std=1.0)
    verify_mel_stats(str(tmp_path), cfg, allow_mismatch=True)  # explicit override


def test_verify_mel_stats_requires_stats_file(tmp_path):
    with pytest.raises(SystemExit, match="stats.json"):
        verify_mel_stats(str(tmp_path), tiny_config())


def test_verify_rejects_stale_fingerprint(tmp_path):
    cfg = tiny_config()
    _write_stats(tmp_path, mean=cfg.audio.mel_mean, std=cfg.audio.mel_std,
                 fingerprint={"sample_rate": 999, "hop_length": 1, "n_mels": 1,
                              "text_vocab_size": 256})
    with pytest.raises(SystemExit, match="fingerprint"):
        verify_mel_stats(str(tmp_path), cfg)
    # Old shards without any fingerprint are also refused.
    _write_stats(tmp_path, mean=cfg.audio.mel_mean, std=cfg.audio.mel_std,
                 fingerprint=None)
    with pytest.raises(SystemExit, match="fingerprint"):
        verify_mel_stats(str(tmp_path), cfg)


# ------------------------------------------------- padded batches / cropping

def test_padded_batch_all_losses_finite_and_masked():
    """Mixed-length batch: padding must not corrupt any loss."""
    from fact.data.synthetic import SyntheticSpeech, collate
    from fact.model import FaCT
    cfg = tiny_config()
    ds_short = SyntheticSpeech(cfg, n_items=1, seconds=0.6, seed=1)
    ds_long = SyntheticSpeech(cfg, n_items=1, seconds=1.4, seed=2)
    batch = collate([ds_short[0], ds_long[0]], cfg)
    # Padded, not truncated: batch length equals the LONG item's length.
    assert batch.mel.shape[1] == batch.mel_lengths.max()
    assert batch.mel_lengths[0] < batch.mel_lengths[1]
    # Padding regions carry the ignore markers.
    s = cfg.encoder.frame_stack
    short_tok = int(batch.mel_lengths[0]) // s
    assert (batch.f0_bins[0, short_tok:] == -100).all()
    torch.manual_seed(0)
    logs = FaCT(cfg).training_step(batch)
    for k, v in logs.items():
        assert torch.isfinite(v), k
    logs["loss"].backward()


def test_crop_drops_transcript(tmp_path):
    """crop_tokens crops audio -> the transcript must be dropped, not kept."""
    from fact.data.dataset import ShardedSpeech
    cfg = tiny_config()
    long_item = {
        "mel": torch.randn(40 * cfg.encoder.frame_stack, cfg.audio.n_mels).to(torch.float16),
        "f0": torch.rand(40 * cfg.encoder.frame_stack) * 200,
        "text": torch.tensor([1, 2, 3], dtype=torch.long),
    }
    short_item = {
        "mel": torch.randn(2 * cfg.encoder.frame_stack, cfg.audio.n_mels).to(torch.float16),
        "f0": torch.rand(2 * cfg.encoder.frame_stack) * 200,
        "text": torch.tensor([4, 5], dtype=torch.long),
    }
    torch.save([long_item, short_item], tmp_path / "shard_000000.pt")
    ds = ShardedSpeech(str(tmp_path), cfg, crop_tokens=8)
    cropped = ds[0]
    assert cropped["mel"].shape[0] == 8 * cfg.encoder.frame_stack
    assert cropped["text"].shape[0] == 0        # transcript dropped
    uncropped = ds[1]
    assert uncropped["text"].shape[0] == 2      # short item keeps its text


def test_ctc_excludes_transcriptless_items():
    """Items with no text contribute nothing to CTC (not pushed to blank)."""
    from fact.modules.heads import CTCHead
    cfg = tiny_config().ctc
    head = CTCHead(16, cfg)
    x = torch.randn(2, 6, 16)
    targets = torch.tensor([[1, 2, 3], [0, 0, 0]])
    lengths = torch.tensor([3, 0])
    loss_mixed, _ = head(x, targets, lengths)
    loss_solo, _ = head(x[:1], targets[:1], lengths[:1])
    assert torch.allclose(loss_mixed, loss_solo, atol=1e-5)
    # All-transcriptless batch: zero loss, still differentiable.
    loss_none, _ = head(x, targets, torch.tensor([0, 0]))
    assert float(loss_none) == 0.0


def test_infeasible_counts_repeat_blanks():
    """'S <= T' alone is not feasibility: adjacent repeats need blanks."""
    from fact.modules.heads import ctc_required_length
    targets = torch.tensor([[1, 1, 2, 2, 3]])   # two adjacent repeats
    req = ctc_required_length(targets, torch.tensor([5]))
    assert int(req) == 7
