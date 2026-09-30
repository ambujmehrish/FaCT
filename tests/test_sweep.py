"""The lambda_ctc sweep runner must stay executable (it gates the pilot)."""

import json
import sys
from pathlib import Path

import torch

from fact.config import tiny_config
from fact.data.synthetic import SyntheticSpeech


def test_sweep_runs_two_arms(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sweep_lambda_ctc", Path("scripts/sweep_lambda_ctc.py")
    )
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)

    cfg = tiny_config()
    ds = SyntheticSpeech(cfg, n_items=8, seconds=1.0)
    items = [{"mel": ds[i]["mel"].to(torch.float16), "f0": ds[i]["f0"],
              "text": ds[i]["text"]} for i in range(8)]
    shards = tmp_path / "shards"
    shards.mkdir()
    torch.save(items, shards / "shard_000000.pt")
    cfg_path = tmp_path / "cfg.yaml"
    cfg.save_yaml(str(cfg_path))

    out = tmp_path / "sweep"
    argv = sys.argv
    sys.argv = ["sweep_lambda_ctc.py", "--config", str(cfg_path),
                "--shards", str(shards), "--lambdas", "0.5,2.0",
                "--steps", "2", "--batch-size", "2", "--eval-utts", "4",
                "--device", "cpu", "--out", str(out)]
    try:
        sweep.main()
    finally:
        sys.argv = argv

    arms = json.loads((out / "sweep_report.json").read_text())
    assert [a["lambda"] for a in arms] == [0.5, 2.0]
    for a in arms:
        assert "greedy_byte_cer" in a and "content_utilization" in a
        assert all(v == v for v in a["final"].values())  # no NaNs
    table = (out / "sweep_table.md").read_text()
    assert "| 0.5 |" in table and "| 2 |" in table
