"""Parallel, resumable Emilia manifest scan."""

import json

from fact.data.manifest import scan_emilia, scan_emilia_parallel, write_manifest


def make_emilia(root, n_dirs=3, n_utts=4):
    for d in range(n_dirs):
        sub = root / f"EN-B{d:06d}"
        sub.mkdir(parents=True)
        for u in range(n_utts):
            stem = f"EN_B{d:05d}_S00000_W{u:06d}"
            (sub / f"{stem}.mp3").write_bytes(b"fake")
            text = "" if u == 0 else f"utt {d} {u}"
            (sub / f"{stem}.json").write_text(json.dumps({"text": text}))
    (root / "EN-B999999.extracting").mkdir()  # half-extracted dir: ignored


def test_parallel_matches_serial(tmp_path):
    root = tmp_path / "EN"
    make_emilia(root)
    serial = tmp_path / "serial.tsv"
    write_manifest(scan_emilia(root), serial, require_text=True)
    par = tmp_path / "par.tsv"
    kept, skipped = scan_emilia_parallel(root, par, workers=2, require_text=True)
    assert (kept, skipped) == (9, 3)
    assert par.read_text() == serial.read_text()
    assert not (tmp_path / "par.tsv.parts").exists()  # cleaned up on success


def test_resumes_from_existing_parts(tmp_path):
    root = tmp_path / "EN"
    make_emilia(root)
    out = tmp_path / "m.tsv"
    parts = tmp_path / "m.tsv.parts"
    parts.mkdir()
    # A part from an earlier, interrupted run is reused, not rescanned.
    (parts / "EN-B000000.tsv").write_text("/prev/run.mp3\tfrom earlier run\n")
    kept, _ = scan_emilia_parallel(root, out, workers=2, require_text=True)
    lines = out.read_text().splitlines()
    assert lines[0] == "/prev/run.mp3\tfrom earlier run"
    assert kept == 1 + 2 * 3


def test_flat_root_falls_back_to_serial(tmp_path):
    root = tmp_path / "flat"
    root.mkdir()
    (root / "a.mp3").write_bytes(b"x")
    (root / "a.json").write_text(json.dumps({"text": "hello"}))
    kept, _ = scan_emilia_parallel(root, tmp_path / "m.tsv", workers=2)
    assert kept == 1
