import torch
from torch.utils.data import DataLoader
from functools import partial

from fact.config import tiny_config
from fact.data.synthetic import SyntheticSpeech, collate
from fact.model import FaCT
from fact.train.trainer import Trainer


def make_batch(cfg, batch_size=2, seconds=1.0):
    ds = SyntheticSpeech(cfg, n_items=batch_size, seconds=seconds)
    return collate([ds[i] for i in range(batch_size)], cfg)


def test_tokenize_detokenize_shapes():
    cfg = tiny_config()
    model = FaCT(cfg).eval()
    batch = make_batch(cfg)
    toks = model.tokenize(mel=batch.mel)
    b, t_tok = toks.content_indices.shape
    assert toks.prosody_indices.shape == (b, t_tok)
    assert toks.content_indices.max() < model.bottleneck.content_codebook_size
    assert toks.residual is not None and toks.residual.shape == (b, t_tok, cfg.bottleneck.residual_dim)

    mel = model.detokenize(
        toks.content_indices, toks.prosody_indices,
        ref_mel=batch.mel, residual=toks.residual, nfe=2,
    )
    assert mel.shape == (b, t_tok * cfg.encoder.frame_stack, cfg.audio.n_mels)


def test_wav_tokenize():
    cfg = tiny_config()
    model = FaCT(cfg).eval()
    wav = torch.randn(2, cfg.audio.sample_rate)  # 1 s
    toks = model.tokenize(wav=wav)
    # ~12.5 tokens/s equivalent at tiny rates: just check nonempty + long dtype.
    assert toks.content_indices.shape[1] > 0
    assert toks.content_indices.dtype == torch.long


def test_training_step_all_losses_finite():
    cfg = tiny_config()
    model = FaCT(cfg)
    batch = make_batch(cfg)
    logs = model.training_step(batch, stage="a")
    for k, v in logs.items():
        assert torch.isfinite(v), f"{k} not finite"
    logs["loss"].backward()
    grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert grads, "encoder got no gradients"


def test_stage_b_freezes_tokenizer():
    cfg = tiny_config()
    model = FaCT(cfg)
    model.freeze_tokenizer()
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(not p.requires_grad for p in model.bottleneck.parameters())
    assert all(p.requires_grad for p in model.decoder.parameters())
    batch = make_batch(cfg)
    logs = model.training_step(batch, stage="b")
    assert torch.isfinite(logs["loss"])


def test_residual_disabled():
    cfg = tiny_config()
    cfg.bottleneck.residual_dim = 0
    model = FaCT(cfg)
    batch = make_batch(cfg)
    toks = model.tokenize(mel=batch.mel)
    assert toks.residual is None
    logs = model.training_step(batch)
    assert torch.isfinite(logs["loss"])
    assert float(logs["kl"]) == 0.0


def test_trainer_smoke():
    cfg = tiny_config()
    model = FaCT(cfg)
    ds = SyntheticSpeech(cfg, n_items=4, seconds=1.0)
    loader = DataLoader(ds, batch_size=2, collate_fn=partial(collate, cfg=cfg))
    trainer = Trainer(model, cfg, stage="a", log_fn=lambda s: None)
    logs = trainer.fit(loader, max_steps=3)
    assert trainer.state.step == 3
    assert all(v == v for v in logs.values())  # no NaNs


def test_param_counts_base_scale():
    """Paper-scale config should land in the plan's target ranges."""
    from fact.config import FaCTConfig
    counts = FaCT(FaCTConfig()).param_counts()
    assert 60e6 < counts["encoder"] < 160e6, counts
    assert 140e6 < counts["decoder"] < 260e6, counts
