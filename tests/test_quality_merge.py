"""--quality-only merge of UTMOS / speaker similarity into result JSONs."""

import importlib.util
import json
from pathlib import Path

import pytest


def _eb():
    spec = importlib.util.spec_from_file_location("eb", Path("scripts/eval_baselines.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_merge_quality_adds_metrics_and_keeps_the_rest(tmp_path):
    eb = _eb()
    old = {"baseline": "x", "aggregate": {"pesq_wb": {"mean": 2.0, "std": 0.1, "n": 2}},
           "per_utterance": {"a": {"pesq_wb": 2.1, "extras": {"wer": 0.1}},
                             "b": {"pesq_wb": 1.9}}}
    path = tmp_path / "x.json"
    path.write_text(json.dumps(old))
    new = {"aggregate": {"utmos": {"mean": 3.5, "std": 0.2, "n": 2}},
           "per_utterance": {"a": {"extras": {"utmos": 3.6, "spk_sim": 0.9}},
                             "b": {"extras": {"utmos": 3.4, "spk_sim": 0.8}}}}
    r = eb.merge_quality(path, new)
    assert r["per_utterance"]["a"]["extras"] == {"wer": 0.1, "utmos": 3.6, "spk_sim": 0.9}
    assert r["per_utterance"]["b"]["extras"]["spk_sim"] == 0.8
    assert r["per_utterance"]["a"]["pesq_wb"] == 2.1
    assert set(r["aggregate"]) == {"pesq_wb", "utmos"}


def test_merge_quality_refuses_other_utterances(tmp_path):
    eb = _eb()
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"aggregate": {}, "per_utterance": {"a": {}}}))
    with pytest.raises(SystemExit):
        eb.merge_quality(path, {"aggregate": {}, "per_utterance": {"b": {"extras": {}}}})
    with pytest.raises(SystemExit):
        eb.merge_quality(tmp_path / "missing.json", {"aggregate": {}, "per_utterance": {}})
