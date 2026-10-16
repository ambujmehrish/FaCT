"""summary.md must be writable on batch nodes with an ASCII locale."""

import importlib.util
import json
import sys
from pathlib import Path


def test_summary_only_rebuilds_from_json(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("eval_baselines", Path("scripts/eval_baselines.py"))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    agg = {"pesq_wb": {"mean": 2.5, "std": 0.3, "n": 2}}
    for name in ("mimi", "dac"):
        (tmp_path / f"{name}.json").write_text(json.dumps({
            "baseline": name, "n_utterances": 2, "aggregate": agg,
            "metadata": {"discrete_bps": 1100.0, "frame_rate_hz": 12.5}}))
    # Simulate an ASCII locale: text writes without an explicit encoding fail.
    real = Path.write_text

    def ascii_default(self, data, encoding=None, *a, **k):
        return real(self, data, encoding or "ascii", *a, **k)
    monkeypatch.setattr(Path, "write_text", ascii_default)
    monkeypatch.setattr(sys, "argv", ["eval_baselines.py", "--baselines", "mimi,dac",
                                      "--eval-dir", str(tmp_path), "--out", str(tmp_path),
                                      "--summary-only"])
    ev.main()
    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "| mimi |" in text and "2.500±0.300" in text
