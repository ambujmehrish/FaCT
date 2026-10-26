"""TTS LM stack: delay pattern, masking, KV-cache equivalence, generation
contract, common-utterance filtering, FaCT tokenization from shards, and a
tiny train + resume run."""

import json

import pytest
import torch

from fact.tts import data as tdata
from fact.tts.model import (EOT, TTSLM, TTSConfig, bos, delayed_targets, eos, pad_id,
                            undelay)


def tiny_cfg(**kw):
    return TTSConfig(vocab_sizes=(11, 7), dim=32, n_layers=2, n_heads=4, max_frames=50,
                     max_text=40, **kw)


def test_delay_pattern_round_trip():
    cfg = tiny_cfg()
    codes = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 1]])
    y = delayed_targets(codes, cfg)
    assert y.shape == (2, 4 + 1 + 1)
    assert y[0].tolist() == [1, 2, 3, 4, eos(11), pad_id(11)]
    assert y[1].tolist() == [pad_id(7), 5, 6, 0, 1, eos(7)]
    assert torch.equal(undelay(y, cfg, 4), codes)


def test_collate_and_loss_masking():
    cfg = tiny_cfg()
    items = [{"codes": torch.randint(0, 7, (2, 5)), "text": torch.tensor([3, 4, 5])},
             {"codes": torch.randint(0, 7, (2, 3)), "text": torch.tensor([9])}]
    b = tdata.collate(items, cfg)
    assert b["text"].shape == (2, 4) and b["text"][1].tolist() == [0, 0, 9, EOT]
    assert b["audio_in"][0, 0, 0] == bos(11) and b["audio_in"][0, 1, 0] == bos(7)
    assert torch.equal(b["audio_in"][0, :, 1:7], b["targets"][0, :, :6])
    assert b["valid"][1].tolist() == [False, False, True, True] + [True] * 5 + [False] * 2
    torch.manual_seed(0)
    m = TTSLM(cfg)
    loss, logs = m.loss(b)
    assert torch.isfinite(loss) and set(logs) == {"ce0", "ce1"}
    # Padding never changes a real position's logits (left text pad, right audio pad).
    one = tdata.collate(items[1:], cfg)
    lb = m(b["text"], b["audio_in"], b["valid"])[0][1, :5]
    l1 = m(one["text"], one["audio_in"], one["valid"])[0][0, :5]
    assert torch.allclose(lb, l1, atol=1e-4)


def test_kv_cache_matches_full_forward():
    cfg = tiny_cfg()
    torch.manual_seed(1)
    m = TTSLM(cfg).eval()
    b = tdata.collate([{"codes": torch.randint(0, 7, (2, 6)), "text": torch.tensor([3, 4, 5, 6])}], cfg)
    full = m(b["text"], b["audio_in"], b["valid"])
    lt = b["text"].shape[1]
    caches = [dict() for _ in m.blocks]
    x = m.embed(b["text"], b["audio_in"][:, :, :1])
    n = x.shape[1]
    pos = torch.arange(n)[None]
    mask = torch.ones(n, n, dtype=torch.bool).tril()[None, None]
    for blk, c in zip(m.blocks, caches):
        x = blk(x, pos, mask, c)
    hs = [m.norm(x[:, -1])]
    for j in range(1, b["audio_in"].shape[2]):
        x = m.embed(b["text"][:, :0], b["audio_in"][:, :, j:j + 1])
        p = torch.tensor([[lt + j]])
        for blk, c in zip(m.blocks, caches):
            x = blk(x, p, None, c)
        hs.append(m.norm(x[:, -1]))
    inc = m.heads[0](torch.stack(hs, 1))
    assert torch.allclose(inc, full[0], atol=1e-4)


def test_generate_contract():
    cfg = tiny_cfg()
    torch.manual_seed(2)
    m = TTSLM(cfg).eval()
    prompt = torch.randint(0, 7, (2, 4))
    g = torch.Generator().manual_seed(0)
    out = m.generate(torch.tensor([3, 4, 5]), prompt, max_new_frames=9, temperature=1.0,
                     top_k=5, generator=g)
    assert out.shape[0] == 2 and 1 <= out.shape[1] <= 9
    assert int(out[0].max()) < 11 and int(out[1].max()) < 7  # codes only, no specials
    g2 = torch.Generator().manual_seed(0)
    again = m.generate(torch.tensor([3, 4, 5]), prompt, max_new_frames=9, temperature=1.0,
                       top_k=5, generator=g2)
    assert torch.equal(out, again)  # reproducible with a seeded generator


def _token_dir(path, codec, rows_per_shard, vocab=(11, 7)):
    path.mkdir(parents=True)
    (path / "meta.json").write_text(json.dumps({"codec": codec, "n_streams": 2,
                                                "vocab_sizes": list(vocab), "frame_rate_hz": 12.5}))
    for name, rows in rows_per_shard.items():
        torch.save(rows, path / name)


def _row(t, n_text=3):
    return {"codes": torch.randint(0, 7, (2, t)).short(), "text": torch.arange(1, n_text + 1).short()}


def test_common_utterances_only(tmp_path):
    cfg = tiny_cfg()
    a, b = tmp_path / "a", tmp_path / "b"
    _token_dir(a, "x", {"shard_000000.pt": [_row(5), _row(6), None], "shard_000001.pt": [_row(4)]})
    _token_dir(b, "y", {"shard_000000.pt": [_row(5), None, _row(3)]})
    assert tdata.common_shards(a, [b]) == ["shard_000000.pt"]
    items = tdata.load_items("shard_000000.pt", a, [b], cfg)
    assert len(items) == 1 and items[0]["codes"].shape == (2, 5)
    ds = tdata.TokenBatches(a, [b], ["shard_000000.pt"], cfg, max_tokens=1000, infinite=False)
    assert sum(x["text"].shape[0] for x in ds) == 1


def test_fact_tokenize_from_shards(tmp_path):
    from fact.config import tiny_config
    from fact.data.preprocess import encode_text_bytes
    from fact.model import FaCT
    from fact.tts import tokenize as tk
    cfg = tiny_config()
    torch.manual_seed(0)
    ck = tmp_path / "fact.pt"
    torch.save({"model": FaCT(cfg).state_dict(), "step": 1, "config": cfg.to_dict()}, ck)
    sh = tmp_path / "shards"
    sh.mkdir()
    items = [{"mel": torch.randn(20 + 4 * i, cfg.audio.n_mels).half(), "f0": torch.zeros(20 + 4 * i),
              "text": encode_text_bytes(f"hi {i}", 257)} for i in range(5)]
    torch.save(items, sh / "shard_000000.pt")
    out = tmp_path / "tok"
    tk.main(["--codec", "fact", "--fact-ckpt", str(ck), "--shards", str(sh), "--out", str(out),
             "--part", "0", "--num-parts", "1", "--device", "cpu", "--batch", "3", "--check-batching", "5"])
    rows = torch.load(out / "shard_000000.pt", weights_only=True)
    assert len(rows) == 5 and rows[2]["codes"].shape == (2, (20 + 8) // cfg.encoder.frame_stack)
    assert torch.equal(rows[1]["text"].long(), items[1]["text"])
    meta = json.loads((out / "meta.json").read_text())
    assert meta["codec"] == "fact" and len(meta["vocab_sizes"]) == 2
    tk.main(["--codec", "fact", "--fact-ckpt", str(ck), "--shards", str(sh), "--out", str(out),
             "--part", "0", "--num-parts", "1", "--device", "cpu"])  # done shards are skipped
    with pytest.raises(SystemExit):
        tk.main(["--codec", "mimi", "--shards", str(sh), "--out", str(tmp_path / "m"),
                 "--num-parts", "1", "--part", "0", "--device", "cpu"])  # no audio index


def test_train_and_resume(tmp_path):
    from fact.tts import train as tt
    a, b = tmp_path / "a", tmp_path / "b"
    rows = {f"shard_{i:06d}.pt": [_row(4 + i), _row(5), _row(3)] for i in range(4)}
    _token_dir(a, "x", rows)
    _token_dir(b, "y", rows)
    common = ["--tokens", str(a), "--require", str(b), "--ckpt-dir", str(tmp_path / "ck"),
              "--dim", "32", "--layers", "1", "--heads", "4", "--max-tokens", "200",
              "--n-val-shards", "1", "--val-every", "2", "--val-batches", "2", "--ckpt-every", "2",
              "--log-every", "1", "--workers", "0", "--device", "cpu", "--warmup", "1"]
    tt.main(common + ["--steps", "2"])
    assert (tmp_path / "ck" / "step_0000002.pt").exists()
    with pytest.raises(SystemExit):  # different settings are refused on resume
        tt.main(common[:-2] + ["--device", "cpu", "--steps", "4", "--lr", "0.1"])


def test_fact_indices_rebuild_the_decoder_condition():
    """The TTS decode path (indices -> embed_indices) gives exactly the
    tokens-only condition stage D was trained on."""
    from fact.config import tiny_config
    from fact.model import FaCT
    cfg = tiny_config()
    torch.manual_seed(0)
    m = FaCT(cfg).eval()
    out = m.encode_mel(torch.randn(1, 24, cfg.audio.n_mels))
    rebuilt = m.bottleneck.embed_indices(out.content_indices, out.prosody_indices, None)
    assert torch.allclose(rebuilt, out.content_emb + out.prosody_emb, atol=1e-5)


def _tiny_tts(tmp_path, monkeypatch):
    """Tiny FaCT + stage-D decoder + untrained tiny LM + 5 LS-style utterances."""
    import importlib.util
    import sys

    import numpy as np
    import soundfile as sf

    from fact.config import tiny_config
    from fact.model import FaCT
    from fact.train import stage_d_wave_decoder as sd
    cfg = tiny_config()
    cfg.bottleneck.residual_capacity = 2.0
    cfg.bottleneck.residual_dropout = 0.5
    torch.manual_seed(0)
    fck = tmp_path / "fact.pt"
    torch.save({"model": FaCT(cfg).state_dict(), "step": 1, "config": cfg.to_dict()}, fck)
    ev = tmp_path / "ls"
    ev.mkdir()
    rng = np.random.default_rng(0)
    lines = []
    for i, (spk, sec) in enumerate([(11, 3.5), (11, 4.5), (22, 3.2), (22, 5.0), (33, 4.2)]):
        p = ev / f"{i:04d}_{spk}-1-{i}.wav"
        sf.write(p, (0.1 * rng.standard_normal(int(16_000 * sec))).astype("float32"), 16_000)
        p.with_suffix(".txt").write_text("HELLO THERE WORLD")
        lines.append(f"{p}\thello\n")
    man = tmp_path / "m.tsv"
    man.write_text("".join(lines))
    monkeypatch.setattr(sys, "argv", ["x", "--tiny", "--fact-ckpt", str(fck), "--manifest", str(man),
                                      "--dev-manifest", str(man), "--ckpt-dir", str(tmp_path / "wd"),
                                      "--device", "cpu", "--steps", "2"])
    sd.main()
    bn = FaCT(cfg).bottleneck
    tcfg = TTSConfig(vocab_sizes=(bn.content_codebook_size, bn.prosody_codebook_size), dim=32,
                     n_layers=1, n_heads=4, max_frames=200, max_text=120)
    lm = tmp_path / "lm.pt"
    torch.save({"model": TTSLM(tcfg).state_dict(), "config": tcfg.to_dict(), "step": 0,
                "run": {"codec_meta": {"codec": "fact", "fact_ckpt": str(fck)}}}, lm)
    return lm, tmp_path / "wd" / "step_0000002.pt", ev


def test_eval_tts_end_to_end(tmp_path, monkeypatch):
    import importlib.util

    from fact.eval import factorization as fz
    lm, dec, ev = _tiny_tts(tmp_path, monkeypatch)
    spec = importlib.util.spec_from_file_location("et", "scripts/eval_tts.py")
    et = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(et)
    assert et.sentence_case("HELLO THERE WORLD") == "Hello there world."
    for shard in (0, 1):
        et.main(["--lm", str(lm), "--wave-decoder", str(dec),
                 "--eval-dir", str(ev), "--out", str(tmp_path / f"t_shard{shard}.json"),
                 "--shard", str(shard), "--num-shards", "2", "--no-quality", "--device", "cpu"])
    res = fz.merge(sorted(str(p) for p in tmp_path.glob("t_shard*.json")), tmp_path / "t.json")
    # Targets 4-10 s: utts 1, 3, 4; utt 4's speaker has no other utterance -> skipped.
    assert res["n_utterances"] == 2 and list(res["skipped"]) == ["0004_33-1-4.wav"]
    one = next(iter(res["per_utterance"].values()))
    assert one["tts/prompt_s"] >= 3.0 and "tts/frames" in one
    with pytest.raises(SystemExit):
        et.parse_args(["--lm", str(lm), "--eval-dir", str(ev), "--out", "x", "--top-p", "0"])


def test_generate_with_fixed_stream0_resamples_only_the_rest():
    cfg = tiny_cfg()
    torch.manual_seed(3)
    m = TTSLM(cfg).eval()
    prompt = torch.randint(0, 7, (2, 3))
    fixed = torch.tensor([1, 5, 9, 2, 0])
    outs = [m.generate(torch.tensor([3, 4]), prompt, temperature=1.0, top_k=0,
                       generator=torch.Generator().manual_seed(s), fixed_stream0=fixed) for s in range(4)]
    for o in outs:
        assert o.shape == (2, 5) and torch.equal(o[0], fixed)  # content kept, same length
    assert len({tuple(o[1].tolist()) for o in outs}) > 1        # the other stream varies
    with pytest.raises(ValueError):
        m.generate(torch.tensor([3]), prompt, fixed_stream0=torch.tensor([11]))


def test_eval_tts_prosody_end_to_end(tmp_path, monkeypatch):
    import importlib.util
    lm, dec, ev = _tiny_tts(tmp_path, monkeypatch)
    spec = importlib.util.spec_from_file_location("ep", "scripts/eval_tts_prosody.py")
    ep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ep)
    out = tmp_path / "p.json"
    ep.main(["--lm", str(lm), "--wave-decoder", str(dec), "--eval-dir", str(ev), "--out", str(out),
             "--resamples", "2", "--device", "cpu"])
    res = json.loads(out.read_text())
    assert set(res["skipped"]) >= {"0004_33-1-4.wav"} and res["per_utterance"]  # not vacuous
    for m in res["per_utterance"].values():
        assert {"fixed/median_st_sd", "free/median_st_sd", "fixed/contour_cents", "fixed/dur_sd_s"} <= set(m)
        assert m["fixed/dur_sd_s"] == 0.0  # content fixed -> identical length


def test_train_require_complete_refuses_partial_token_sets(tmp_path):
    from fact.tts import train as tt
    a, b = tmp_path / "a", tmp_path / "b"
    _token_dir(a, "x", {f"shard_{i:06d}.pt": [_row(4)] for i in range(3)})
    _token_dir(b, "y", {f"shard_{i:06d}.pt": [_row(4)] for i in range(2)})  # one shard missing
    with pytest.raises(SystemExit, match="different shard sets"):
        tt.main(["--tokens", str(a), "--require", str(b), "--ckpt-dir", str(tmp_path / "ck"),
                 "--require-complete", "--n-val-shards", "1", "--device", "cpu", "--steps", "1"])
