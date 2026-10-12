"""The offline-readiness verifier must stay importable and strict."""

import importlib.util
import json
import sys
from pathlib import Path


def _load():
    spec = importlib.util.spec_from_file_location("verify_env", Path("scripts/verify_env.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check_paths_flags_home(monkeypatch, tmp_path):
    ve = _load()
    ve.RESULTS.clear()
    home = tmp_path / "home"
    outside = tmp_path / "work"
    for d in (home, outside):
        d.mkdir()
    monkeypatch.setenv("FACT_BASE", str(outside))
    monkeypatch.setenv("FACT_CACHE", str(home / "cache"))  # the violation
    monkeypatch.setenv("HF_HOME", str(outside / "hf"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(outside / "xdg"))
    monkeypatch.setenv("PIP_CACHE_DIR", str(outside / "pip"))
    monkeypatch.setenv("TORCH_HOME", str(outside / "torch"))
    ve.check_paths(home.resolve())
    by_name = {n: l for l, n, _ in ve.RESULTS}
    assert by_name["env FACT_CACHE"] == "FAIL"   # inside $HOME -> hard fail
    assert by_name["env HF_HOME"] == "PASS"
    assert by_name["env FACT_BASE"] == "PASS"


def test_check_paths_flags_unset(monkeypatch, tmp_path):
    ve = _load()
    ve.RESULTS.clear()
    for var in ["FACT_BASE", "FACT_CACHE", "HF_HOME", "XDG_CACHE_HOME",
                "PIP_CACHE_DIR", "TORCH_HOME"]:
        monkeypatch.delenv(var, raising=False)
    ve.check_paths((tmp_path / "home").resolve())
    assert all(l == "FAIL" for l, n, _ in ve.RESULTS if n.startswith("env "))


def test_verifier_cli_fails_on_empty_base(tmp_path, monkeypatch):
    """End-to-end: empty base dir -> FAILs -> exit 1, report written."""
    ve = _load()
    ve.RESULTS.clear()
    base = tmp_path / "base"
    base.mkdir()
    monkeypatch.setenv("FACT_BASE", str(base))
    monkeypatch.setenv("FACT_CACHE", str(base / "fact_cache"))
    monkeypatch.setenv("HF_HOME", str(base / "hf"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(base / "xdg"))
    monkeypatch.setenv("PIP_CACHE_DIR", str(base / "pip"))
    monkeypatch.setenv("TORCH_HOME", str(base / "torch"))
    out = base / "report.json"
    argv = sys.argv
    sys.argv = ["verify_env.py", "--base", str(base), "--out", str(out)]
    try:
        import pytest
        with pytest.raises(SystemExit):
            ve.main()  # models/data absent -> must exit nonzero
    finally:
        sys.argv = argv
    report = json.loads(out.read_text())
    assert report["fail"] > 0
    names = {r["name"] for r in report["results"]}
    assert "DAC weights in $FACT_CACHE" in names
    assert "eval set staged" in names
    # Core imports must pass in this environment.
    import_results = [r for r in report["results"] if r["name"] == "import torch"]
    assert import_results and import_results[0]["level"] == "PASS"
