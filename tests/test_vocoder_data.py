"""RandomSegments manifest sampling: one manifest draws as before; several
comma-separated manifests draw as their concatenation."""

import random

import pytest

from fact.vocoder.data import RandomSegments


def _manifest(path, prefix, n):
    path.write_text("".join(f"/{prefix}/{i:04d}.wav\ttext {i:04d}\n" for i in range(n)))
    return path


def _draws(ds, n, seed=0):
    rng = random.Random(seed)
    fhs = [open(m, "rb") for m in ds.manifests]
    try:
        return [ds._draw_path(fhs, rng) for _ in range(n)]
    finally:
        for fh in fhs:
            fh.close()


def test_two_manifests_draw_like_their_concatenation(tmp_path):
    a, b = _manifest(tmp_path / "a.tsv", "a", 300), _manifest(tmp_path / "b.tsv", "b", 100)
    both = RandomSegments(f"{a},{b}", 16_000, 100, seed=0)
    assert both.shares() == pytest.approx({str(a): 0.75, str(b): 0.25})
    got = _draws(both, 4000)
    frac_b = sum(p.startswith("/b/") for p in got) / len(got)
    assert 0.21 < frac_b < 0.29 and {p[:3] for p in got} == {"/a/", "/b/"}
    # every draw is a whole first field of one manifest line
    assert all(p.endswith(".wav") and len(p) == len("/a/0000.wav") for p in got)


def test_single_manifest_draws_are_unchanged(tmp_path):
    a = _manifest(tmp_path / "a.tsv", "a", 300)
    ds = RandomSegments(str(a), 16_000, 100, seed=0)
    rng, want = random.Random(0), []
    with open(a, "rb") as fh:  # the original single-file procedure
        while len(want) < 200:
            fh.seek(rng.randrange(a.stat().st_size))
            fh.readline()
            line = fh.readline()
            if line.strip():
                want.append(line.split(b"\t", 1)[0].decode().strip())
    assert _draws(ds, 200) == want


def test_empty_manifest_is_refused(tmp_path):
    a = _manifest(tmp_path / "a.tsv", "a", 3)
    (tmp_path / "e.tsv").write_text("")
    with pytest.raises(ValueError, match="empty manifest"):
        RandomSegments(f"{a},{tmp_path / 'e.tsv'}", 16_000, 100, seed=0)
