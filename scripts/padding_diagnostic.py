"""Padding diagnostic (audit 2026-11-03, C): how much did attending to batch
padding during training affect a model, and what does it cost at evaluation?

    python scripts/padding_diagnostic.py --ckpt <step.pt> --label <name> \
        --shards <held-out dev shards> --out results/padding_diagnostic

Batches are built exactly as in training (fact.train.stage_a.build_loader:
same sampler, seed, batch size and collate), from held-out dev shards. On
each batch the model's own training losses and tokens are computed twice
with the same random draws:

  legacy  the pre-fix behaviour (padding visible), i.e. what the model was
          trained under;
  fixed   padding hidden, i.e. every item as if it were alone, which is the
          evaluation condition (eval encodes one utterance at a time).

fixed - legacy > 0 on a loss means the model does worse under the evaluation
condition than under its training condition: the reported numbers of a
model trained with the bug are then conservative. Token agreement says how
much padding changed the tokens themselves. A first few batches are also
encoded item by item to confirm that `fixed` equals encoding alone.

Run the causal main model as a control: strictly causal attention never sees
the padded tail, so its legacy and fixed results should (nearly) agree.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fact.config import FaCTConfig  # noqa: E402
from fact.eval.grounding import token_lengths, valid_indices  # noqa: E402
from fact.model import FaCT  # noqa: E402
from fact.modules.transformer import legacy_unmasked_padding  # noqa: E402


def to_device(batch, device, ssl_fn):
    from fact.model import Batch

    def mv(x):
        return x.to(device) if isinstance(x, torch.Tensor) else x
    out = Batch(mel=mv(batch.mel), text=mv(batch.text), text_lengths=mv(batch.text_lengths),
                f0_bins=mv(batch.f0_bins), energy=mv(batch.energy), ref_mel=mv(batch.ref_mel),
                mel_lengths=mv(batch.mel_lengths), wav16=mv(batch.wav16),
                wav_lengths=mv(batch.wav_lengths))
    if ssl_fn is not None:
        if out.wav16 is None:
            raise SystemExit("SSL-input model but the batch has no audio (shards need an audio_index)")
        out.ssl = ssl_fn(out.wav16, out.wav_lengths)
    return out


def _losses(model, batch, seed):
    torch.manual_seed(seed)
    logs = model.training_step(batch)
    return {k: float(v) for k, v in logs.items() if torch.is_tensor(v) and v.numel() == 1}


def _tokens(model, batch):
    out = model.encode_mel(batch.mel, batch.ssl, lengths=batch.mel_lengths)
    return out


def _agree(a, b, lengths, r=1):
    if a is None:
        return None
    va, vb = valid_indices(a, lengths, r), valid_indices(b, lengths, r)
    return float((va == vb).float().mean())


def _rel_l2(a, b, lengths):
    if a is None:
        return None
    num = den = 0.0
    for i, n in enumerate(lengths.tolist()):
        num += float((a[i, :n] - b[i, :n]).pow(2).sum())
        den += float(b[i, :n].pow(2).sum())
    return (num / max(den, 1e-12)) ** 0.5


def paired_ci(diff: np.ndarray, n_boot: int = 2000, seed: int = 0) -> list[float]:
    """95% bootstrap CI of the mean over batches (the resampling unit)."""
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


@torch.no_grad()
def diagnose(model: FaCT, loader, n_batches: int, alone_batches: int, device, ssl_fn) -> dict:
    s = model.cfg.encoder.frame_stack
    r = model.bottleneck.prosody_rate_divisor
    rows, alone = [], []
    for bi, raw in enumerate(loader):
        if bi >= n_batches:
            break
        batch = to_device(raw, device, ssl_fn)
        lengths_mel = batch.mel_lengths.cpu()
        t = batch.mel.shape[1]
        pad_frac = 1.0 - float(lengths_mel.sum()) / (len(lengths_mel) * t)
        with legacy_unmasked_padding():
            loss_l = _losses(model, batch, seed=bi)
            tok_l = _tokens(model, batch)
        loss_f = _losses(model, batch, seed=bi)
        tok_f = _tokens(model, batch)
        tl = token_lengths(batch, tok_f.content_indices.shape[1] if tok_f.content_indices is not None
                           else tok_f.primary_emb().shape[1], s)
        rows.append({
            "pad_frac": pad_frac, "batch": len(lengths_mel), "frames": t,
            "legacy": loss_l, "fixed": loss_f,
            "content_agree": _agree(tok_l.content_indices, tok_f.content_indices, tl),
            "prosody_agree": _agree(tok_l.prosody_indices, tok_f.prosody_indices, tl, r),
            "residual_rel_l2": _rel_l2(tok_l.residual_emb, tok_f.residual_emb, tl)
            if tok_f.residual_emb is not None else None,
        })
        if bi < alone_batches and tok_f.content_indices is not None:
            for i, n in enumerate(lengths_mel.tolist()):
                n = n // s * s
                ssl_i = batch.ssl[i: i + 1, :n] if batch.ssl is not None else None
                one = model.encode_mel(batch.mel[i: i + 1, :n], ssl_i)
                k = n // s
                alone.append(float((one.content_indices[0, :k].cpu()
                                    == tok_f.content_indices[i, :k].cpu()).float().mean()))
        if bi % 20 == 0:
            print(f"batch {bi}: pad {pad_frac:.2f} recon legacy {loss_l.get('recon', float('nan')):.4f} "
                  f"fixed {loss_f.get('recon', float('nan')):.4f} content agree {rows[-1]['content_agree']}",
                  flush=True)
    if not rows:
        raise SystemExit("no batches")
    keys = sorted(set(rows[0]["legacy"]) & set(rows[0]["fixed"]))
    losses = {}
    for k in keys:
        a = np.array([x["legacy"][k] for x in rows])
        b = np.array([x["fixed"][k] for x in rows])
        losses[k] = {"legacy": float(a.mean()), "fixed": float(b.mean()),
                     "fixed_minus_legacy": float((b - a).mean()), "ci95": paired_ci(b - a)}
    pad = np.array([x["pad_frac"] for x in rows])

    def mean_of(key):
        v = [x[key] for x in rows if x[key] is not None]
        return float(np.mean(v)) if v else None
    by_pad = {}
    for lo, hi in ((0.0, 0.1), (0.1, 0.3), (0.3, 1.01)):
        sel = [x for x in rows if lo <= x["pad_frac"] < hi]
        if sel and "recon" in keys:
            by_pad[f"{lo:.1f}-{min(hi, 1.0):.1f}"] = {
                "n_batches": len(sel),
                "recon_fixed_minus_legacy": float(np.mean([x["fixed"]["recon"] - x["legacy"]["recon"]
                                                           for x in sel])),
                "content_agree": float(np.mean([x["content_agree"] for x in sel
                                                if x["content_agree"] is not None] or [np.nan]))}
    return {
        "n_batches": len(rows),
        "padding": {"mean_frac": float(pad.mean()), "median_frac": float(np.median(pad)),
                    "p90_frac": float(np.percentile(pad, 90))},
        "losses": losses,
        "tokens_legacy_vs_fixed": {"content_agree": mean_of("content_agree"),
                                   "prosody_agree": mean_of("prosody_agree"),
                                   "residual_rel_l2": mean_of("residual_rel_l2")},
        "fixed_vs_alone_content_agree": float(np.mean(alone)) if alone else None,
        "n_alone_items": len(alone),
        "by_padding": by_pad,
        "batches": rows,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--shards", required=True, help="held-out shards (e.g. <data>/shards/dev)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-batches", type=int, default=200)
    ap.add_argument("--alone-batches", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=None, help="default: the run's training batch")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    if not Path(a.shards).is_dir():
        raise SystemExit(f"{a.shards}: not a directory")

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    cfg = FaCTConfig.from_dict(ck["config"])
    run = ck.get("run") or {}
    bs = a.batch_size or run.get("batch_size")
    if not bs:
        raise SystemExit(f"{a.ckpt} records no batch_size: pass --batch-size")
    model = FaCT(cfg)
    model.load_state_dict(ck["model"])
    model = model.to(a.device).eval()
    step = ck.get("step")
    del ck

    audio_index = None
    if cfg.encoder.ssl_model:
        audio_index = str(Path(a.shards) / "audio_index")
        if not Path(audio_index).is_dir():
            raise SystemExit(f"{cfg.encoder.ssl_model} model needs {audio_index}")
    from fact.modules.ssl import ssl_for_config
    from fact.train.stage_a import build_loader
    ssl_fn = ssl_for_config(cfg, a.device)
    loader, sampler = build_loader(SimpleNamespace(synthetic=False, shards=a.shards, crop_tokens=None,
                                                   audio_index=audio_index, batch_size=bs,
                                                   num_workers=a.num_workers), cfg)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)
    print(f"{a.label}: {a.ckpt} step {step}, batch {bs}, encoder causal={cfg.encoder.causal}, "
          f"decoder causal={cfg.decoder.causal}", flush=True)
    rep = diagnose(model, loader, a.n_batches, a.alone_batches, a.device, ssl_fn)
    from fact.provenance import provenance
    rep = {"label": a.label, "ckpt": a.ckpt, "step": step, "shards": a.shards, "batch_size": bs,
           "encoder_causal": cfg.encoder.causal, "decoder_causal": cfg.decoder.causal, **rep,
           "provenance": provenance()}
    a.out.mkdir(parents=True, exist_ok=True)
    dst = a.out / f"{a.label}.json"
    tmp = dst.with_name(f"{dst.name}.{os.getpid()}.partial")
    tmp.write_text(json.dumps(rep, indent=1))
    tmp.replace(dst)
    print(json.dumps({k: rep[k] for k in ("padding", "losses", "tokens_legacy_vs_fixed",
                                          "fixed_vs_alone_content_agree")}, indent=1))


if __name__ == "__main__":
    main()
