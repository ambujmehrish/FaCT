"""--match-run: a retrain takes the reference run's config and batch size and
refuses to start when any recorded run setting differs."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from fact.train.stage_a import check_match_run

ROOT = Path(__file__).resolve().parents[1]


def test_check_match_run_accepts_identical_and_refuses_differences():
    ref = {"world_size": 1, "shards": "/s", "max_steps": 10, "grad_accum": 1, "batch_size": 4}
    check_match_run(ref, {"world_size": 1, "shards": "/s", "match_run": "x"},
                    batch_size=4, max_steps=10, grad_accum=1)
    with pytest.raises(SystemExit, match="batch_size"):
        check_match_run(ref, {"world_size": 1, "shards": "/s"}, batch_size=8, max_steps=10, grad_accum=1)
    with pytest.raises(SystemExit, match="init_from"):  # recorded but not set here: refused
        check_match_run({**ref, "init_from": "a.pt"}, {"world_size": 1, "shards": "/s"},
                        batch_size=4, max_steps=10, grad_accum=1)


def _run(args, tmp):
    return subprocess.run([sys.executable, "-m", "fact.train.stage_a", "--synthetic", "--device", "cpu",
                           *args], cwd=ROOT, capture_output=True, text=True, timeout=600)


def test_match_run_cli_reuses_config_and_batch(tmp_path):
    ref_dir, new_dir = tmp_path / "ref", tmp_path / "new"
    r = _run(["--tiny", "--steps", "2", "--ckpt-every", "2", "--batch-size", "2",
              "--ckpt-dir", str(ref_dir)], tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    ref = sorted(ref_dir.glob("step_*.pt"))[-1]
    r = _run(["--match-run", str(ref), "--steps", "2", "--ckpt-every", "2", "--ckpt-dir", str(new_dir)],
             tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    a = torch.load(ref, weights_only=True)
    b = torch.load(sorted(new_dir.glob("step_*.pt"))[-1], weights_only=True)
    assert a["config"] == b["config"]
    assert b["run"]["batch_size"] == 2 and b["run"]["match_run"] == str(ref)
    r = _run(["--match-run", str(ref), "--steps", "2", "--batch-size", "3", "--ckpt-dir",
              str(tmp_path / "bad")], tmp_path)
    assert r.returncode != 0 and "batch_size" in (r.stderr + r.stdout)
    r = _run(["--match-run", str(ref), "--config", "configs/fact_base.yaml", "--ckpt-dir",
              str(tmp_path / "bad2")], tmp_path)
    assert r.returncode != 0
