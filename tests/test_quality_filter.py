"""Clean-subset manifests from Emilia DNSMOS sidecars."""

import json

import pytest

from fact.data import quality_filter as qf


def _corpus(tmp_path, scores):
    lines = []
    for i, s in enumerate(scores):
        a = tmp_path / f"u{i}.mp3"
        a.write_bytes(b"")
        meta = {"text": "x", "duration": 3600.0} if s is None else {"text": "x", "dnsmos": s, "duration": 3600.0}
        a.with_suffix(".json").write_text(json.dumps(meta))
        lines.append(f"{a}\ttext {i}\n")
    m = tmp_path / "train.tsv"
    m.write_text("".join(lines))
    return m


def test_scan_merge_thresholds(tmp_path):
    m = _corpus(tmp_path, [2.9, 3.25, 3.45, 3.55, 3.7, 3.1])
    out = tmp_path / "q"
    for k in range(2):
        qf.scan(m, out, k, 2, threads=2)
    rep = qf.merge(out, 2, thresholds=(3.0, 3.4, 3.5))
    assert rep["kept"]["3.4"] == {"utterances": 3, "hours": 3.0}
    assert rep["kept"]["3.0"]["utterances"] == 5 and rep["total_hours"] == 6.0
    kept = (out / "train_dnsmos3.5.tsv").read_text().splitlines()
    assert len(kept) == 2 and all(ln.count("\t") == 1 for ln in kept)  # original manifest lines


def test_missing_score_is_an_error_not_a_guess(tmp_path):
    m = _corpus(tmp_path, [3.5, None])
    out = tmp_path / "q"
    rep = qf.scan(m, out, 0, 1, threads=1)
    assert rep["errors"] == 1 and "no 'dnsmos'" in next(iter(rep["error_examples"].values()))
    with pytest.raises(SystemExit):  # 1 of 2 unreadable: refused
        qf.merge(out, 1)
    with pytest.raises(SystemExit):  # a missing part is refused
        qf.merge(out, 2)
