"""Semantic distillation into the content tokens (distill.*) and
fine-tuning from another run's weights (--init-from)."""

from functools import partial

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import fact.modules.ssl as ssl_mod
from fact.config import FaCTConfig, tiny_config
from fact.model import FaCT
from fact.modules.ssl import SSLFeatures
from tests.test_ssl_input import _shards_with_audio, _tiny_wavlm


def _cfg():
    cfg = tiny_config()
    cfg.distill.weight, cfg.distill.dim = 1.0, 32
    return cfg


def test_distill_loss_requires_teacher_and_trains_content_path():
    from fact.data.synthetic import SyntheticSpeech, collate
    cfg = _cfg()
    model = FaCT(cfg)
    ds = SyntheticSpeech(cfg, n_items=2, seconds=1.0)
    b = collate([ds[0], ds[1]], cfg)
    with pytest.raises(ValueError, match="teacher"):
        model.training_step(b)
    b.teacher = torch.randn(2, b.mel.shape[1], 32)
    logs = model.training_step(b)
    assert 0 <= float(logs["distill"]) <= 2 and torch.isfinite(logs["loss"])
    logs["loss"].backward()
    assert model.distill_head.weight.grad is not None
    # The gradient reaches the encoder through the quantized content tokens.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())


def test_distill_target_is_mean_over_each_tokens_frames():
    cfg = _cfg()
    model = FaCT(cfg)
    s = cfg.encoder.frame_stack
    with torch.no_grad():
        model.distill_head.weight.copy_(torch.eye(32, cfg.encoder.dim))
        model.distill_head.bias.zero_()
    emb = torch.randn(1, 3, cfg.encoder.dim)
    # Teacher frames whose per-token mean equals the first 32 dims of emb.
    teacher = emb[..., :32].repeat_interleave(s, dim=1)
    teacher = torch.cat([teacher, torch.randn(1, 2, 32)], 1)  # extra trailing frames ignored
    mask = torch.ones(1, 3, dtype=torch.bool)
    assert float(model._distill_loss(emb, teacher, mask, s)) < 1e-5
    with pytest.raises(ValueError, match="frames"):
        model._distill_loss(emb, teacher[:, : 3 * s - 1], mask, s)


def test_distill_off_by_default_and_old_configs_load():
    cfg = tiny_config()
    assert cfg.distill.weight == 0 and FaCT(cfg).distill_head is None
    d = cfg.to_dict()
    del d["distill"]  # checkpoints written before distill existed
    assert FaCTConfig.from_dict(d).distill.weight == 0


def test_trainer_distills_and_init_from(tmp_path, monkeypatch):
    from fact.data.dataset import ShardedSpeech, collate
    from fact.train.trainer import Trainer
    cfg, sh, _ = _shards_with_audio(tmp_path)
    cfg.train.max_steps, cfg.train.log_every = 2, 1
    real = SSLFeatures
    monkeypatch.setattr(ssl_mod, "SSLFeatures", lambda model_id, layer: real(layer=2, model=_tiny_wavlm()))
    ds = ShardedSpeech(str(sh), cfg, audio_index=str(sh / "audio_index"))
    loader = DataLoader(ds, batch_size=2, collate_fn=partial(collate, cfg=cfg), drop_last=True)

    base = Trainer(FaCT(cfg), cfg, device="cpu", ckpt_dir=str(tmp_path / "base"))
    base.fit(loader, max_steps=2)
    base_ck = tmp_path / "base.pt"
    base.save(base_ck)

    ft_cfg = FaCTConfig.from_dict(cfg.to_dict())
    ft_cfg.distill.weight, ft_cfg.distill.dim = 1.0, 32
    tr = Trainer(FaCT(ft_cfg), ft_cfg, device="cpu", ckpt_dir=str(tmp_path / "ft"))
    new = tr.init_from(str(base_ck))
    assert {k.split(".")[0] for k in new} == {"distill_head"} and tr.state.step == 0
    sd = torch.load(base_ck, weights_only=True)["model"]
    now = tr.model.state_dict()
    assert all(torch.equal(now[k], v) for k, v in sd.items())
    assert tr.teacher is not None
    logs = tr.fit(loader, max_steps=2)
    assert tr.state.step == 2 and np.isfinite(logs["distill"])

    # A checkpoint of a different architecture is refused.
    other = tiny_config()
    other.encoder.dim = 32
    bad = tmp_path / "bad.pt"
    torch.save({"model": FaCT(other).state_dict(), "step": 1}, bad)
    with pytest.raises((SystemExit, RuntimeError)):
        tr.init_from(str(bad))


def test_select_wer_setting_rule(tmp_path, monkeypatch, capsys):
    import importlib.util
    import json
    import sys
    from pathlib import Path

    def write(t, g, wer_err, mel):
        per = {f"u{i}": {"mel_l1": 1.0, "extras": {"mel_l1_band": mel, "wer_errors": wer_err,
                                                   "wer_words": 10.0, "utmos": 3.0}}
               for i in range(5)}
        (tmp_path / f"fact_r@0400000_full_T{t}_G{g}.json").write_text(json.dumps({"per_utterance": per}))
    write("0.5", "1.5", 2.0, 1.10)   # reference: WER 20 %
    write("0.3", "2.0", 1.0, 1.11)   # better WER, mel within slack -> selected
    write("0.0", "3.0", 0.5, 1.20)   # best WER but mel-L1 over the limit
    spec = importlib.util.spec_from_file_location("s", Path("scripts/select_wer_setting.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(sys, "argv", ["x", "--run", "r", "--dir", str(tmp_path)])
    m.main()
    assert capsys.readouterr().out.split() == ["0.3", "32", "2.0"]
