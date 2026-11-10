"""scripts/backup_scratch.py: copies only what is under the scratch root,
mirrors paths, rewrites manifests, reports missing items, and is resumable."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("backup_scratch", ROOT / "scripts/backup_scratch.py")
bs = importlib.util.module_from_spec(spec)
sys.modules["backup_scratch"] = bs
spec.loader.exec_module(bs)


def test_backup_copies_scratch_only_and_rewrites_manifest(tmp_path):
    scratch, work = tmp_path / "scratch", tmp_path / "work"
    runs = scratch / "runs"
    (runs / "a").mkdir(parents=True)
    (runs / "a" / "step_0000010.pt").write_bytes(b"x" * 100)
    (runs / "a" / "config.yaml").write_text("k: 1\n")
    audio_s = scratch / "emilia" / "u1.mp3"
    audio_s.parent.mkdir(parents=True)
    audio_s.write_bytes(b"a" * 10)
    audio_w = work / "ls" / "u2.wav"
    audio_w.parent.mkdir(parents=True)
    audio_w.write_bytes(b"b" * 10)
    man = scratch / "manifests" / "test.tsv"
    man.parent.mkdir(parents=True)
    man.write_text(f"{audio_s}\thello\n{audio_w}\tworld\n")
    shards = scratch / "shards" / "dev"
    shards.mkdir(parents=True)
    (shards / "s0.pt").write_bytes(b"s")
    lst = tmp_path / "list.txt"
    lst.write_text("# comment\na/step_0000010.pt\nb/step_0000020.pt\n")
    out = work / "backup"
    args = ["--out", str(out), "--scratch-root", str(scratch), "--runs-dir", str(runs),
            "--checkpoints", str(lst), "--manifest", str(man), "--dir", str(shards)]
    assert bs.main(args) == 1  # b/step_0000020.pt is missing
    copied = out / "scratch" / "runs" / "a" / "step_0000010.pt"
    assert copied.read_bytes() == b"x" * 100
    assert (out / "scratch" / "runs" / "a" / "config.yaml").exists()
    assert (out / "scratch" / "emilia" / "u1.mp3").exists()
    assert (out / "scratch" / "shards" / "dev" / "s0.pt").exists()
    rows = (out / "manifests" / "test.tsv").read_text().splitlines()
    assert rows[0].split("\t") == [str(out / "scratch" / "emilia" / "u1.mp3"), "hello"]
    assert rows[1].split("\t") == [str(audio_w), "world"]  # already on WORK: unchanged
    rep = json.loads((out / "backup_manifest.json").read_text())
    assert rep["missing"] == [str(runs / "b" / "step_0000020.pt")]
    assert str(audio_w) in rep["already_safe"]
    ck = [c for c in rep["copied"] if c["source"].endswith("step_0000010.pt")][0]
    assert len(ck["sha256"]) == 64
    # resumable: a second run copies nothing new
    bs.main(args)
    rep2 = json.loads((out / "backup_manifest.json").read_text())
    assert rep2["copied"] == [] and rep2["skipped_existing"] >= 4


def test_backup_refuses_output_on_scratch(tmp_path):
    import pytest
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    lst = tmp_path / "l.txt"
    lst.write_text("")
    with pytest.raises(SystemExit, match="on scratch"):
        bs.main(["--out", str(scratch / "o"), "--scratch-root", str(scratch), "--runs-dir",
                 str(scratch), "--checkpoints", str(lst)])
