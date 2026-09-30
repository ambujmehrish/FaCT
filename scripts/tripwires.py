"""Gate tripwires: RISK_ASSESSMENT R1/R2 metrics from a checkpoint.

One command per pilot checkpoint; the week-6 gate reads the JSON:

    python scripts/tripwires.py --config <cfg.yaml> --ckpt <step.pt> \
        --shards <dir> --out tripwires.json [--device cuda]

Reports:
  R1 (factorization collapse):
    - per-stream codebook utilization (fraction of codes used) and
      perplexity (exp of index entropy = effective codebook size)
    - quick probes: content->F0 vs prosody->F0 accuracy (content must be
      the WORSE F0 predictor; parity with prosody = leak)
  R2 (residual dominance):
    - reconstruction-loss deltas when zeroing one channel at a time,
      with fixed noise/timestep so deltas are comparable: if zeroing the
      residual hurts >> zeroing prosody/content, the continuous channel
      is carrying the signal and the discrete-interface claims are void.

Thresholds (from RISK_ASSESSMENT SS2) are evaluated into pass/warn flags;
they are gates for HUMAN review, not auto-aborts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fact.config import FaCTConfig
from fact.data.dataset import ShardedSpeech, collate
from fact.eval.leakage import probe_accuracy
from fact.model import FaCT


def stream_stats(indices: torch.Tensor, codebook_size: int) -> dict:
    counts = torch.bincount(indices.flatten(), minlength=codebook_size).float()
    probs = counts / counts.sum().clamp_min(1)
    nz = probs[probs > 0]
    entropy = float(-(nz * nz.log()).sum())
    return {
        "codebook_size": codebook_size,
        "codes_used": int((counts > 0).sum()),
        "utilization": float((counts > 0).float().mean()),
        "perplexity": float(math.exp(entropy)),
    }


@torch.no_grad()
def channel_zeroing_deltas(model: FaCT, batch, device: str, n_draws: int = 8) -> dict:
    """Flow-matching loss under channel ablations, same noise/t draws."""
    out = model.encode_mel(batch.mel.to(device))
    t_tok = out.primary_emb().shape[1]
    s = model.cfg.encoder.frame_stack
    mel_target = model._norm_mel(batch.mel.to(device)[:, : t_tok * s])
    spk = model.speaker_encoder(model._norm_mel(batch.mel.to(device)),
                                lengths=batch.mel_lengths.to(device))

    zero = torch.zeros(())

    def cond_variant(drop: str) -> torch.Tensor:
        parts = {
            "content": out.content_emb,
            "prosody": out.prosody_emb,
            "residual": out.residual_emb,
        }
        acc = None
        for name, emb in parts.items():
            if emb is None or name == drop:
                continue
            acc = emb if acc is None else acc + emb
        return acc

    results = {}
    b = mel_target.shape[0]
    for drop in ["none", "content", "prosody", "residual"]:
        cond = cond_variant(drop if drop != "none" else "")
        if cond is None:
            continue
        losses = []
        for i in range(n_draws):
            g = torch.Generator(device="cpu").manual_seed(1234 + i)
            t = torch.rand(b, generator=g).to(device)
            noise = torch.randn(mel_target.shape, generator=g).to(device)
            x_t = (1 - t.view(b, 1, 1)) * noise + t.view(b, 1, 1) * mel_target
            v = model.decoder(x_t, t, cond, spk)
            losses.append(float(((v - (mel_target - noise)) ** 2).mean()))
        results[drop] = sum(losses) / len(losses)
    deltas = {k: results[k] - results["none"] for k in results if k != "none"}
    return {"loss": results, "delta_vs_full": deltas}


def quick_f0_probes(model: FaCT, batches: list, device: str) -> dict:
    """content->F0 vs prosody->F0 accuracy on frozen features (valid frames)."""
    feats = {"content": [], "prosody": []}
    targets = []
    with torch.no_grad():
        for batch in batches:
            out = model.encode_mel(batch.mel.to(device))
            t_tok = out.primary_emb().shape[1]
            f0 = batch.f0_bins.to(device)[:, :t_tok]
            valid = f0 >= 0
            if out.content_emb is not None:
                feats["content"].append(out.content_emb[valid].cpu())
            if out.prosody_emb is not None:
                feats["prosody"].append(out.prosody_emb[valid].cpu())
            targets.append(f0[valid].cpu())
    y = torch.cat(targets)
    n_classes = model.cfg.prosody.f0_bins + 1
    accs = {}
    for name, chunks in feats.items():
        if not chunks:
            continue
        x = torch.cat(chunks)

        def make_batches(x=x, y=y):
            for i in range(0, x.shape[0], 2048):
                yield x[i : i + 2048].unsqueeze(0), y[i : i + 2048].unsqueeze(0)

        accs[f"{name}_to_f0_acc"] = probe_accuracy(
            make_batches, in_dim=x.shape[-1], n_classes=n_classes, steps=300
        )
    return accs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-utts", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = FaCTConfig.from_yaml(args.config)
    model = FaCT(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model = model.to(args.device).eval()

    ds = ShardedSpeech(args.shards, cfg)
    idxs = list(range(0, len(ds), max(1, len(ds) // args.n_utts)))[: args.n_utts]
    batches = [collate([ds[i] for i in idxs[j : j + 4]], cfg)
               for j in range(0, len(idxs), 4)]

    report: dict = {"ckpt": args.ckpt, "step": ckpt["step"]}

    toks = [model.tokenize(mel=b.mel.to(args.device)) for b in batches]
    if toks[0].content_indices is not None:
        report["content"] = stream_stats(
            torch.cat([t.content_indices.cpu().flatten() for t in toks]),
            model.bottleneck.content_codebook_size,
        )
    if toks[0].prosody_indices is not None:
        report["prosody"] = stream_stats(
            torch.cat([t.prosody_indices.cpu().flatten() for t in toks]),
            model.bottleneck.prosody_codebook_size,
        )
    report["channel_zeroing"] = channel_zeroing_deltas(model, batches[0], args.device)
    report["probes"] = quick_f0_probes(model, batches, args.device)

    # RISK SS2 gate flags.
    flags = {}
    if "prosody" in report:
        flags["r1_prosody_underused"] = report["prosody"]["perplexity"] < 50
    p_acc = report["probes"].get("prosody_to_f0_acc")
    c_acc = report["probes"].get("content_to_f0_acc")
    if p_acc is not None and c_acc is not None:
        flags["r1_content_leaks_f0"] = c_acc > p_acc - 0.05
    d = report["channel_zeroing"]["delta_vs_full"]
    if "residual" in d and "prosody" in d:
        flags["r2_residual_dominant"] = d["residual"] > 3 * max(d["prosody"], 1e-6)
    report["flags"] = flags

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
    if any(flags.values()):
        print("\nWARN: tripwire(s) fired - see RISK_ASSESSMENT SS2 for pivots.")


if __name__ == "__main__":
    main()
