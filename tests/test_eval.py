import torch

from fact.config import tiny_config
from fact.data.synthetic import SyntheticSpeech, collate
from fact.eval.leakage import probe_accuracy
from fact.eval.modelability import TokenLMProbe, evaluate_bits, train_probe
from fact.eval.reconstruction import (
    chunk_boundary_discontinuity, f0_metrics, reconstruction_metrics,
)
from fact.model import FaCT


def test_modelability_probe_bits():
    torch.manual_seed(0)
    vocabs = [50, 10]
    probe = TokenLMProbe(vocabs, dim=32, n_layers=1, n_heads=4)
    batches = [
        [torch.randint(0, v, (2, 16)) for v in vocabs] for _ in range(3)
    ]
    train_probe(probe, batches, steps=5)
    res = evaluate_bits(probe, batches, frame_rate_hz=12.5)
    # Random streams: bits/frame near log2(50) + log2(10) ~ 8.96.
    assert 0 < res.bits_per_frame < 12
    assert abs(res.bits_per_second - res.bits_per_frame * 12.5) < 1e-6
    assert set(res.breakdown) == {"stream_0", "stream_1"}


def test_modelability_learns_structure():
    """A constant stream must cost ~0 bits after a few steps."""
    torch.manual_seed(0)
    probe = TokenLMProbe([7], dim=32, n_layers=1, n_heads=4)
    batches = [[torch.full((4, 12), 3, dtype=torch.long)] for _ in range(2)]
    train_probe(probe, batches, steps=60, lr=1e-2)
    res = evaluate_bits(probe, batches, frame_rate_hz=12.5)
    assert res.bits_per_frame < 0.2


def test_leakage_probe_separates_signal_from_noise():
    torch.manual_seed(0)
    n_classes = 4

    def informative():
        for _ in range(8):
            target = torch.randint(0, n_classes, (2, 16))
            feats = torch.nn.functional.one_hot(target, n_classes).float()
            feats = torch.cat([feats, torch.randn(2, 16, 4)], dim=-1)
            yield feats, target

    def uninformative():
        g = torch.Generator().manual_seed(1)
        for _ in range(8):
            target = torch.randint(0, n_classes, (2, 16), generator=g)
            yield torch.randn(2, 16, 8, generator=g), target

    acc_hi = probe_accuracy(informative, in_dim=8, n_classes=n_classes, steps=200)
    acc_lo = probe_accuracy(uninformative, in_dim=8, n_classes=n_classes, steps=200)
    assert acc_hi > 0.9
    assert acc_lo < 0.6


def test_reconstruction_metrics():
    torch.manual_seed(0)
    ref = torch.randn(1, 50, 20)
    hyp = ref + 0.01 * torch.randn_like(ref)
    m = reconstruction_metrics(ref, hyp)
    assert m.mel_l1 < 0.05
    assert m.energy_corr > 0.95

    f0_ref = torch.tensor([[0.0, 100.0, 110.0, 0.0, 120.0]])
    f0_hyp = torch.tensor([[0.0, 105.0, 110.0, 115.0, 120.0]])
    rmse, f1 = f0_metrics(f0_ref, f0_hyp)
    assert 0 < rmse < 10
    assert 0 < f1 <= 1.0


def test_chunk_discontinuity_smooth_signal():
    t = torch.linspace(0, 6.28, 64)
    mel = torch.sin(t).view(1, 64, 1).expand(1, 64, 8)
    d = chunk_boundary_discontinuity(mel, block_size=8)
    assert 0.5 < d["ratio"] < 2.0  # no seams in a smooth signal


def test_end_to_end_eval_on_model_tokens():
    """Tokenize synthetic speech, then probe the actual token streams."""
    cfg = tiny_config()
    model = FaCT(cfg).eval()
    ds = SyntheticSpeech(cfg, n_items=4, seconds=1.0)
    batch = collate([ds[i] for i in range(4)], cfg)
    toks = model.tokenize(mel=batch.mel)
    vocabs = [model.bottleneck.content_codebook_size,
              model.bottleneck.prosody_codebook_size]
    probe = TokenLMProbe(vocabs, dim=32, n_layers=1, n_heads=4)
    stream_batches = [[toks.content_indices, toks.prosody_indices]]
    train_probe(probe, stream_batches, steps=3)
    res = evaluate_bits(probe, stream_batches, frame_rate_hz=cfg.token_rate_hz)
    assert res.bits_per_frame > 0
