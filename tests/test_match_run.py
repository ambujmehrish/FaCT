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


def test_config_diff_dotted_keys():
    from fact.train.stage_a import config_diff
    a = {"bottleneck": {"variant": "factorized", "levels": [8, 4]}, "encoder": {"dim": 8}}
    assert config_diff(a, a) == []
    b = {"bottleneck": {"variant": "vae", "levels": (8, 4)}, "encoder": {"dim": 8}}
    assert config_diff(a, b) == ["bottleneck.variant"]  # tuple vs list is not a change


def test_ablation_of_changes_only_the_bottleneck(tmp_path):
    ref_dir = tmp_path / "ref"
    r = _run(["--tiny", "--steps", "2", "--ckpt-every", "2", "--batch-size", "2",
              "--ckpt-dir", str(ref_dir)], tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    ref = sorted(ref_dir.glob("step_*.pt"))[-1]
    new_dir = tmp_path / "abl"
    r = _run(["--ablation-of", str(ref), "--set", "bottleneck.variant=single_fsq", "--steps", "2",
              "--ckpt-every", "2", "--ckpt-dir", str(new_dir)], tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    a = torch.load(ref, weights_only=True)
    b = torch.load(sorted(new_dir.glob("step_*.pt"))[-1], weights_only=True)
    assert b["config"]["bottleneck"]["variant"] == "single_fsq"
    assert {k: v for k, v in b["config"].items() if k != "bottleneck"} == \
        {k: v for k, v in a["config"].items() if k != "bottleneck"}
    assert b["run"]["batch_size"] == 2 and b["run"]["ablation_of"] == str(ref)
    for bad in (["--set", "encoder.dim=16"], [], ["--set", "bottleneck.variant=factorized"]):
        r = _run(["--ablation-of", str(ref), *bad, "--steps", "2", "--ckpt-dir",
                  str(tmp_path / "bad")], tmp_path)
        assert r.returncode != 0, bad


def test_ablation_of_with_allow_change_trains_another_token_rate(tmp_path):
    ref_dir = tmp_path / "ref"
    r = _run(["--tiny", "--steps", "2", "--ckpt-every", "2", "--batch-size", "2",
              "--ckpt-dir", str(ref_dir)], tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    ref = sorted(ref_dir.glob("step_*.pt"))[-1]
    sets = ["--set", "encoder.frame_stack=2", "--set", "bottleneck.residual_capacity=1.0"]
    allow = ["--allow-change", "encoder.frame_stack", "--allow-change", "bottleneck.residual_capacity"]
    r = _run(["--ablation-of", str(ref), *sets, *allow, "--steps", "2", "--ckpt-every", "2",
              "--ckpt-dir", str(tmp_path / "hz25")], tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    a = torch.load(ref, weights_only=True)["config"]
    b = torch.load(sorted((tmp_path / "hz25").glob("step_*.pt"))[-1], weights_only=True)["config"]
    assert b["encoder"]["frame_stack"] == 2 and a["encoder"]["frame_stack"] == 4
    from fact.train.stage_a import config_diff
    assert sorted(config_diff(a, b)) == ["bottleneck.residual_capacity", "encoder.frame_stack"]
    # refused: the change is outside the default scope, outside the given scope, or no reference
    for bad in ([*sets], ["--set", "encoder.dim=16", *allow], [*sets, "--allow-change", "encoder.frame_stack"]):
        r = _run(["--ablation-of", str(ref), *bad, "--steps", "2", "--ckpt-dir", str(tmp_path / "bad")], tmp_path)
        assert r.returncode != 0, bad
    r = _run(["--tiny", *allow, "--steps", "2", "--ckpt-dir", str(tmp_path / "bad2")], tmp_path)
    assert r.returncode != 0


def test_same_global_batch_accepts_another_gpu_layout_only():
    ref = {"world_size": 8, "shards": "/s", "max_steps": 10, "grad_accum": 1, "batch_size": 8}
    one_node = {"world_size": 4, "shards": "/s"}
    check_match_run(ref, one_node, same_global_batch=True, batch_size=8, max_steps=10, grad_accum=2)
    check_match_run(ref, one_node, same_global_batch=True, batch_size=16, max_steps=10, grad_accum=1)
    with pytest.raises(SystemExit, match="world_size"):  # not accepted without the flag
        check_match_run(ref, one_node, batch_size=8, max_steps=10, grad_accum=2)
    with pytest.raises(SystemExit, match="samples per optimizer step"):  # half the batch
        check_match_run(ref, one_node, same_global_batch=True, batch_size=8, max_steps=10, grad_accum=1)
    with pytest.raises(SystemExit, match="shards"):  # every other setting still has to match
        check_match_run(ref, {"world_size": 4, "shards": "/other"}, same_global_batch=True,
                        batch_size=8, max_steps=10, grad_accum=2)


def test_ablation_of_on_another_layout_cli(tmp_path):
    ref_dir = tmp_path / "ref"
    r = _run(["--tiny", "--steps", "2", "--ckpt-every", "2", "--batch-size", "2",
              "--ckpt-dir", str(ref_dir)], tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    ref = sorted(ref_dir.glob("step_*.pt"))[-1]
    base = ["--ablation-of", str(ref), "--set", "encoder.frame_stack=2", "--set", "train.grad_accum=2",
            "--allow-change", "encoder.frame_stack", "--allow-change", "train.grad_accum",
            "--steps", "2", "--ckpt-every", "2"]
    r = _run([*base, "--same-global-batch", "--batch-size", "1", "--ckpt-dir", str(tmp_path / "ok")], tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    run = torch.load(sorted((tmp_path / "ok").glob("step_*.pt"))[-1], weights_only=True)["run"]
    assert run["batch_size"] == 1 and run["grad_accum"] == 2
    # same layout change without the flag, and a doubled batch with it: both refused
    r = _run([*base, "--batch-size", "1", "--ckpt-dir", str(tmp_path / "bad")], tmp_path)
    assert r.returncode != 0
    r = _run([*base, "--same-global-batch", "--ckpt-dir", str(tmp_path / "bad2")], tmp_path)
    assert r.returncode != 0 and "samples per optimizer step" in (r.stderr + r.stdout)
