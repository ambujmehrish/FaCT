import torch

from fact.config import tiny_config
from fact.data.synthetic import SyntheticSpeech, collate
from fact.model import FaCT


def _batch(cfg, n=2):
    ds = SyntheticSpeech(cfg, n_items=n, seconds=1.0)
    return collate([ds[i] for i in range(n)], cfg)


def test_shortcut_loss_finite_and_backprops():
    cfg = tiny_config()
    model = FaCT(cfg)
    batch = _batch(cfg)
    out = model.encode_mel(batch.mel)
    spk = model.speaker_encoder(batch.mel)
    t_tok = out.content_emb.shape[1]
    mel = batch.mel[:, : t_tok * cfg.encoder.frame_stack]
    loss = model.decoder.shortcut_loss(mel, out.decoder_condition(), spk, fm_fraction=0.5)
    assert torch.isfinite(loss)
    loss.backward()


def test_sample_nfe_variants():
    cfg = tiny_config()
    model = FaCT(cfg).eval()
    cond = torch.randn(1, 6, cfg.encoder.dim)
    spk = torch.randn(1, cfg.speaker.dim)
    for nfe in (1, 2, 4):
        mel = model.decoder.sample(cond, spk, nfe=nfe)
        assert mel.shape == (1, 6 * cfg.encoder.frame_stack, cfg.audio.n_mels)
        assert torch.isfinite(mel).all()


def test_cfg_guidance_runs():
    cfg = tiny_config()
    model = FaCT(cfg).eval()
    cond = torch.randn(1, 4, cfg.encoder.dim)
    spk = torch.randn(1, cfg.speaker.dim)
    mel = model.decoder.sample(cond, spk, nfe=2, cfg_scale=2.0)
    assert torch.isfinite(mel).all()


def test_d_to_id_mapping():
    cfg = tiny_config()
    model = FaCT(cfg)
    dec = model.decoder
    ids = dec.d_to_id(torch.tensor([0.0, 1.0, 0.5, 0.25, 2.0 ** -cfg.decoder.max_shortcut_log2]))
    assert ids[0] == 0                     # d=0 -> plain FM slot
    assert ids[1] == cfg.decoder.max_shortcut_log2 + 1  # d=1 -> largest slot
    assert ids[4] == 1                     # smallest supported step
    assert (ids[1] > ids[2] > ids[3]).item()


def test_prosody_swap_interface():
    """RQ2 demo path: content from A + prosody from B decodes to valid mel."""
    cfg = tiny_config()
    model = FaCT(cfg).eval()
    batch = _batch(cfg, n=2)
    toks = model.tokenize(mel=batch.mel)
    mel = model.detokenize(
        toks.content_indices[:1],
        toks.prosody_indices[1:2],  # swapped stream
        ref_mel=batch.mel[:1],
        nfe=1,
    )
    assert torch.isfinite(mel).all()
