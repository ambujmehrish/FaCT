"""Registry + wrapper import surface (no checkpoints downloaded in CI)."""

import pytest

from fact.baselines import base as registry


def test_registry_lists_wrappers():
    names = registry.available()
    assert "mimi" in names and "dac" in names


def test_unknown_baseline_raises():
    with pytest.raises(KeyError):
        registry.build("nope", device="cpu")


def test_eval_script_importable():
    """The CLI must import (argparse wiring, metric plumbing) without GPUs."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "eval_baselines", Path("scripts/eval_baselines.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "eval_baseline") and hasattr(mod, "write_summary")


def test_write_summary_format(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "eval_baselines", Path("scripts/eval_baselines.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fake = {
        "baseline": "mimi",
        "metadata": {"frame_rate_hz": 12.5, "discrete_bps": 1100.0},
        "n_utterances": 3,
        "aggregate": {"pesq_wb": {"mean": 2.2, "std": 0.1, "n": 3}},
    }
    text = mod.write_summary([fake], tmp_path)
    assert "mimi" in text and "1.10" in text and "2.200" in text
    assert (tmp_path / "summary.md").exists()
