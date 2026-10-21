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
    fact = {"bitrate": {"discrete_bps": 278.6, "residual_bps": 144.3},
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
    assert "FaCT main (tokens only)" in text and "0.279" in text
    assert "FaCT main (tokens + residual)" in text and "0.423" in text
    assert "n/a" in text                           # FaCT has no waveform metrics yet
    assert "-0." in text and "*" in text           # paired delta, significant
