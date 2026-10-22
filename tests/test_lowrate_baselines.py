"""Low-bitrate codec wrappers: safe checkpoint loading, frame-rate probe,
registration and the missing-source error. The real models are exercised
on the cluster (they need their cloned source and weights)."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from fact.baselines import base as registry
from fact.baselines import lowrate


def test_registered():
    assert {"focalcodec", "wavtokenizer", "bigcodec"} <= set(registry.available())


def test_missing_source_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv("FACT_CODEC_SRC", str(tmp_path))
    with pytest.raises(ImportError, match="fetch_lowrate_codecs"):
        lowrate.codec_source("BigCodec")


def test_load_checkpoint_is_weights_only(tmp_path, monkeypatch):
    path = tmp_path / "ok.pt"
    torch.save({"state_dict": {"w": torch.ones(2)}, "epoch": 3}, path)
    assert torch.equal(lowrate.load_checkpoint(str(path))["state_dict"]["w"], torch.ones(2))
    # A pickled custom object is refused, never unpickled.
    mod = types.ModuleType("fake_framework_mod")

    class Obj:
        pass
    Obj.__module__, Obj.__qualname__ = "fake_framework_mod", "Obj"
    mod.Obj = Obj
    monkeypatch.setitem(sys.modules, "fake_framework_mod", mod)
    bad = tmp_path / "bad.pt"
    torch.save({"x": Obj()}, bad)
    with pytest.raises(Exception, match="(?i)weights only|weights_only"):
        lowrate.load_checkpoint(str(bad))


def test_measured_frame_rate():
    enc = lambda x: torch.zeros(1, x.shape[-1] // 320 + 1)  # 50 Hz at 16 kHz, +1 edge frame
    assert lowrate.measured_frame_rate(enc, 16_000, "cpu") == 50.0
    enc = lambda x: torch.zeros(1, x.shape[-1] // 1280)  # 12.5 Hz
    assert lowrate.measured_frame_rate(enc, 16_000, "cpu") == 12.5


def test_spec_label_is_a_filename():
    spec = importlib.util.spec_from_file_location("eb", Path("scripts/eval_baselines.py"))
    eb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eb)
    assert eb.spec_label("focalcodec:config=lucadellalib/focalcodec_50hz") == \
        "focalcodec_configlucadellalib_focalcodec_50hz"
