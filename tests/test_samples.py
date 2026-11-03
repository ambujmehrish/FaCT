"""Listening-sample rendering (scripts/render_codec_samples.py) and the
static page (scripts/make_sample_page.py), with a dummy codec."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _ls_dir(tmp_path, sr=16_000):
    import soundfile as sf
    d = tmp_path / "ls"
    d.mkdir()
    rng = np.random.default_rng(0)
    # speaker 11: two utterances (prompt partner); speaker 22: one (no partner, dropped)
    for i, (name, sec) in enumerate([("0000_11-1-0000", 5.0), ("0001_11-1-0001", 3.5), ("0002_22-2-0000", 6.0)]):
        sf.write(d / f"{name}.wav", (0.1 * rng.standard_normal(int(sec * sr))).astype("float32"), sr)
        (d / f"{name}.txt").write_text(f"TEXT {i}")
    return d


def test_render_and_page(tmp_path, monkeypatch):
    from fact.baselines import base as registry

    class Halve:
        sample_rate = 24_000

        def reconstruct(self, wav):
            return 0.5 * wav

    monkeypatch.setitem(registry._REGISTRY, "dummy", lambda device="cpu": Halve())
    rc, pg = _load("render_codec_samples"), _load("make_sample_page")
    out = tmp_path / "s"
    rc.main(["--eval-dir", str(_ls_dir(tmp_path)), "--n", "8", "--out", str(out), "--device", "cpu",
             "--spec", "dummy:label=Dummy_codec"])
    meta = json.loads((out / "utterances.json").read_text())
    # Only targets of 4-10 s with a same-speaker prompt of >= 3 s, as in eval_tts.py.
    assert [u["stem"] for u in meta["utterances"]] == ["0000_11-1-0000"]
    assert meta["utterances"][0]["prompt_stem"] == "0001_11-1-0001"
    import soundfile as sf
    a, sr = sf.read(out / "codec" / "original" / "0000_11-1-0000.wav")
    b, _ = sf.read(out / "codec" / "Dummy_codec" / "0000_11-1-0000.wav")
    assert sr == 16_000 and len(a) == len(b)
    # Level matching undoes the dummy codec's gain: both at -26 dBFS RMS.
    for x in (a, b):
        assert abs(20 * np.log10(np.sqrt(np.mean(x ** 2))) + 26.0) < 0.1
    # TTS / prosody folders as eval_tts*.py --save-audio write them.
    (out / "tts" / "LM").mkdir(parents=True)
    sf.write(out / "tts" / "LM" / "0000_11-1-0000.wav", b, sr)
    (out / "prosody" / "LM" / "0000_11-1-0000").mkdir(parents=True)
    for n in ("base", "fixed_1", "free_1"):
        sf.write(out / "prosody" / "LM" / "0000_11-1-0000" / f"{n}.wav", b, sr)
    pg.main(["--dir", str(out)])
    page = (out / "index.html").read_text()
    for rel in ("codec/original/0000_11-1-0000.wav", "codec/Dummy_codec/0000_11-1-0000.wav",
                "tts/LM/0000_11-1-0000.wav", "prosody/LM/0000_11-1-0000/fixed_1.wav"):
        assert f'src="{rel}"' in page and (out / rel).exists()
    assert page.index("<th>original</th>") < page.index("<th>Dummy_codec</th>")


def test_level_never_clips():
    rc = _load("render_codec_samples")
    w = torch.zeros(1000)
    w[10] = 1.0  # one spike: RMS scaling alone would push it far above 1
    assert rc.level(w, -26.0).abs().max() <= 0.99 + 1e-6


def test_provenance_block_names_the_commit():
    import subprocess

    from fact.provenance import provenance
    p = provenance()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert p["git_commit"] == head and isinstance(p["git_dirty"], bool) and p["argv"]
