"""Group-disjoint Emilia split: no held-out group may reach train."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load():
    spec = importlib.util.spec_from_file_location("split_emilia", Path("scripts/split_emilia.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _manifest(path: Path, n_groups=400, per=6):
    lines = []
    for b in range(4):
        for s in range(n_groups // 4):
            for w in range(per):
                lines.append(f"/r/EN-B{b:06d}/EN_B{b:05d}_S{s:05d}_W{w:06d}.mp3\ttext {b} {s} {w}\n")
    path.write_text("".join(lines), encoding="utf-8")
    return len(lines)


def test_split_is_group_disjoint_and_chunked(tmp_path, monkeypatch):
    sp = _load()
    n = _manifest(tmp_path / "all.tsv")
    out = tmp_path / "m"
    monkeypatch.setattr(sys, "argv", ["split_emilia.py", "--manifest", str(tmp_path / "all.tsv"),
                                      "--out-dir", str(out), "--heldout-per-10k", "2000",
                                      "--n-dev", "20", "--n-test", "30", "--per-group", "2",
                                      "--chunks", "3"])
    sp.main()
    g = lambda f: {sp.group_id(ln.split("\t")[0]) for ln in (out / f).read_text().splitlines()}
    train, held, dev, test = g("train.tsv"), g("heldout.tsv"), g("dev.tsv"), g("test.tsv")
    assert not train & held and dev <= held and test <= held and not dev & test
    assert len((out / "test.tsv").read_text().splitlines()) == 30
    chunk_lines = sum(len((out / f"train_c{i:03d}.tsv").read_text().splitlines()) for i in range(3))
    held_lines = len((out / "heldout.tsv").read_text().splitlines())
    assert chunk_lines + held_lines == n
    rep = json.loads((out / "split_report.json").read_text())
    assert rep["train_heldout_group_overlap"] == 0 and rep["dev_test_group_overlap"] == 0
    # Deterministic: same input -> same held-out groups on any machine.
    assert sp.stable_bucket("EN_B00001_S00002") == sp.stable_bucket("EN_B00001_S00002")
    with pytest.raises(SystemExit):  # never overwrite an existing split
        sp.main()


def test_group_id_rejects_non_emilia_names():
    sp = _load()
    assert sp.group_id("/x/EN_B00054_S01894_W000010.mp3") == "EN_B00054_S01894"
    with pytest.raises(ValueError):
        sp.group_id("/x/some_other_file.wav")
