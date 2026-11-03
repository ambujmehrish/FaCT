"""scripts/archive_artifacts.py: inference weights + settings kept, optimizer
dropped, checksums recorded, a missing checkpoint is an error."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch


def _load():
    spec = importlib.util.spec_from_file_location("arch", Path("scripts/archive_artifacts.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_archive_strips_and_checksums(tmp_path):
    arch = _load()
    runs = tmp_path / "runs"
    (runs / "a").mkdir(parents=True)
    w = {"w": torch.randn(4, 4)}
    torch.save({"model": w, "opt": {"state": torch.randn(100)}, "config": {"x": 1}, "step": 7},
               runs / "a" / "step_0000007.pt")
    lst = tmp_path / "list.txt"
    lst.write_text("# comment\na/step_0000007.pt\n")
    out = tmp_path / "arch"
    arch.main(["--runs-dir", str(runs), "--out", str(out), "--list", str(lst)])
    m = json.loads((out / "manifest.json").read_text())
    f = m["files"][0]
    assert f["kept"] == ["config", "model", "step"] and f["dropped"] == ["opt"]
    assert f["archived_sha256"] == arch.sha256(out / "checkpoints" / "a" / "step_0000007.pt")
    ck = torch.load(out / "checkpoints" / "a" / "step_0000007.pt", weights_only=True)
    assert torch.equal(ck["model"]["w"], w["w"]) and "opt" not in ck
    assert (out / "env" / "pip_freeze.txt").read_text().strip() and m["provenance"]["git_commit"]
    lst.write_text("a/step_0000007.pt\nmissing/step_1.pt\n")
    with pytest.raises(SystemExit, match="missing"):
        arch.main(["--runs-dir", str(runs), "--out", str(tmp_path / "arch2"), "--list", str(lst)])
