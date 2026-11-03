"""Reconstruction table: CIs, corpus WER, paired deltas, FaCT rows marked n/a."""

import importlib.util
import json
import sys
from pathlib import Path

from fact.config import FaCTConfig
from fact.eval.reconstruction import band_limited_bins


def _load():
    spec = importlib.util.spec_from_file_location("make_recon_table", Path("scripts/make_recon_table.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_band_limited_bins_cover_source_bandwidth():
    a = FaCTConfig().audio
    assert band_limited_bins(a, 24_000) == a.n_mels       # full band at native rate
    assert 80 <= band_limited_bins(a, 16_000) < a.n_mels  # 16 kHz source: bins <= 8 kHz


def test_table_from_baseline_and_fact_json(tmp_path, monkeypatch):
    utts = [f"u{i}.wav" for i in range(20)]
    base = {"baseline": "mimi", "metadata": {"discrete_bps": 1100.0},
            "per_utterance": {u: {"pesq_wb": 2.0 + 0.01 * i, "stoi": 0.9, "mel_l1": 1.0,
                                  "extras": {"mel_l1_band": 1.0 + 0.01 * i,
                                             "wer_errors": 1.0, "wer_words": 10.0}}
                              for i, u in enumerate(utts)}}
    (tmp_path / "mimi.json").write_text(json.dumps(base))
    fact = {"bitrate": {"discrete_bps": 278.6, "residual_bps": 144.3}, "spk_ref": "other",
            "views": {"full": {}, "tokens_prior": {}},
            "per_utterance": {u: {"full/mel_l1_band": 0.8, "full/energy_corr": 0.99,
                                  "tokens_prior/mel_l1_band": 0.9,
                                  "tokens_prior/energy_corr": 0.98} for u in utts}}
    (tmp_path / "fact.json").write_text(json.dumps(fact))
    out = tmp_path / "t.md"
    mt = _load()
    monkeypatch.setattr(sys, "argv", ["make_recon_table.py", "--baselines", str(tmp_path / "mimi.json"),
                                      "--fact", f"main={tmp_path / 'fact.json'}",
                                      "--title", "T", "--out", str(out)])
    mt.main()
    text = out.read_text()
    assert "| mimi | 1.100 |" in text
    assert "10.00 [" in text                       # corpus WER 10/100 words -> 10%
    assert "FaCT main (tokens only; speaker ref: other)" in text and "| 0.279 |" in text
    # The residual is shown as a capacity bound next to the token bitrate.
    assert "FaCT main (tokens + residual; speaker ref: other)" in text and "0.279 + ≤0.144" in text
    assert "n/a" in text                           # FaCT has no waveform metrics yet
    assert "-0." in text and "*" in text           # paired delta, significant


def _baseline(path, name, sr, vocab, rate, wer_err, utmos, n=30):
    per = {f"u{i}.wav": {"pesq_wb": 2.0, "stoi": 0.9, "mel_l1": 1.0,
                         "extras": {"mel_l1_band": 1.0 + 0.01 * i, "energy_corr": 0.9,
                                    "wer_errors": float(wer_err if i % 2 else 0), "wer_words": 10.0,
                                    "utmos": utmos + 0.01 * i, "spk_sim": 0.9}} for i in range(n)}
    bps = rate * sum(__import__("math").log2(v) for v in vocab)
    path.write_text(json.dumps({"baseline": name, "per_utterance": per,
                                "metadata": {"discrete_bps": bps + (144.3 if name.startswith("factwave") else 0),
                                             "sample_rate": sr, "vocab_sizes": vocab, "frame_rate_hz": rate}}))


def test_oracle_fact_rows_refused(tmp_path):
    import pytest
    mt = _load()
    oracle = {"bitrate": {"discrete_bps": 278.6, "residual_bps": 144.3}, "spk_ref": "full",
              "views": {"full": {}}, "per_utterance": {"u0.wav": {"full/mel_l1_band": 1.0}}}
    (tmp_path / "f.json").write_text(json.dumps(oracle))
    with pytest.raises(SystemExit, match="oracle"):
        mt.main(["--fact", f"m={tmp_path / 'f.json'}", "--title", "T", "--out", str(tmp_path / "t.md")])


def test_band_sensitive_cells_and_paired_table(tmp_path):
    mt = _load()
    _baseline(tmp_path / "low.json", "lowrate16k", 16_000, [8192], 25.0, 2, 4.0)
    _baseline(tmp_path / "fw.json", "factwave_x_full_spkother", 24_000, [8192, 625], 12.5, 1, 3.5)
    out = tmp_path / "t.md"
    mt.main(["--baselines", str(tmp_path / "low.json"), str(tmp_path / "fw.json"), "--title", "T",
             "--out", str(out), "--source-sr", "24000", "--paired", "factwave_x_full_spkother"])
    text = out.read_text()
    low = next(line for line in text.splitlines() if line.startswith("| lowrate16k | 0.325 |"))
    assert low.count("n/c") == 2                         # mel-L1 and energy r only
    assert "0.279 + ≤0.144" in text                     # residual shown as a bound
    pt = text[text.index("**Paired differences, factwave_x_full_spkother"):]
    row = next(line for line in pt.splitlines() if line.startswith("| lowrate16k"))
    assert "-5.00 [" in row                              # corpus WER 5% - 10%, paired
    assert "-0.500 [" in row and "*" in row              # UTMOS 3.5 - 4.0, significant
    assert "n/c" in row                                  # mel-L1 vs a 16 kHz system


def test_cell_ci_independent_of_other_rows(tmp_path):
    """Per-cell seeds: a row's CI is the same whatever else is in the table."""
    mt = _load()
    _baseline(tmp_path / "a.json", "sysA", 24_000, [2048, 2048], 12.5, 3, 3.0)
    _baseline(tmp_path / "b.json", "sysB", 24_000, [2048], 50.0, 1, 3.2)
    one, two = tmp_path / "1.md", tmp_path / "2.md"
    mt.main(["--baselines", str(tmp_path / "a.json"), "--title", "T", "--out", str(one)])
    mt.main(["--baselines", str(tmp_path / "b.json"), str(tmp_path / "a.json"), "--title", "T", "--out", str(two)])
    row = lambda p: next(line for line in p.read_text().splitlines() if line.startswith("| sysA |"))
    assert row(one) == row(two)
