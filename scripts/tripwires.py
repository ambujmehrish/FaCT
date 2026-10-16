"""Gate tripwires: RISK_ASSESSMENT R1/R2/R3 metrics from a checkpoint.

One command per pilot checkpoint; the week-6 gate reads the JSON:

    python scripts/tripwires.py --config <cfg.yaml> --ckpt <step.pt> \
        --shards <dir> --out tripwires.json [--device cuda]

Reports:
  R1 (factorization collapse):
    - per-stream codebook utilization (fraction of codes used) and
      perplexity (exp of index entropy = effective codebook size)
    - quick probes: content->F0 vs prosody->F0 accuracy on held-out
      utterances (content must be the WORSE F0 predictor; parity with
      prosody = leak), next to the majority-bin baseline
  R2 (residual dominance):
    - reconstruction-loss deltas when zeroing one channel at a time,
      with fixed noise/timestep so deltas are comparable: if zeroing the
      residual hurts >> zeroing prosody/content, the continuous channel
      is carrying the signal and the discrete-interface claims are void.
  R3 (grounding): greedy-CTC byte-CER of the content stream.

All metrics use valid (unpadded) frames only. Default 256 utterances:
the content codebook has 8192 codes, and utilization is bounded by
tokens / 8192, so small samples under-report it.

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
from fact.eval.grounding import codebook_stats, greedy_byte_cer, token_lengths, valid_indices
from fact.eval.leakage import probe_accuracy
from fact.model import FaCT


@torch.no_grad()
def channel_zeroing_deltas(model: FaCT, batches: list, device: str,
                           n_draws: int = 8) -> dict:
    """Flow-matching loss under channel ablations, same noise/t draws.

    Averaged over every valid (unpadded) mel frame of all batches.
    """
    s = model.cfg.encoder.frame_stack
    sse: dict[str, float] = {}
    n_frames = 0
    for bi, batch in enumerate(batches):
        out = model.encode_mel(batch.mel.to(device))
        t_tok = out.primary_emb().shape[1]
        mel_target = model._norm_mel(batch.mel.to(device)[:, : t_tok * s])
        spk = model.speaker_encoder(model._norm_mel(batch.mel.to(device)),
                                    lengths=batch.mel_lengths.to(device))
        lengths = token_lengths(batch, t_tok, s).to(device)
        mask = (torch.arange(t_tok, device=device).unsqueeze(0)
                < lengths.unsqueeze(1)).repeat_interleave(s, dim=1)
        n_frames += int(mask.sum()) * n_draws
        parts = {"content": out.content_emb, "prosody": out.prosody_emb,
                 "residual": out.residual_emb}
        b = mel_target.shape[0]
        for drop in ["none", "content", "prosody", "residual"]:
            if drop != "none" and parts[drop] is None:
                continue
            kept = [e for name, e in parts.items() if e is not None and name != drop]
            if not kept:
                continue
            cond = kept[0]
            for e in kept[1:]:
                cond = cond + e
            for i in range(n_draws):
                g = torch.Generator(device="cpu").manual_seed(1234 + 1000 * bi + i)
                t = torch.rand(b, generator=g).to(device)
                noise = torch.randn(mel_target.shape, generator=g).to(device)
                x_t = (1 - t.view(b, 1, 1)) * noise + t.view(b, 1, 1) * mel_target
                v = model.decoder(x_t, t, cond, spk)
                err = ((v - (mel_target - noise)) ** 2).mean(dim=-1)  # (B, T_mel)
                sse[drop] = sse.get(drop, 0.0) + float(err[mask].sum())
    results = {k: v / max(1, n_frames) for k, v in sse.items()}
    deltas = {k: results[k] - results["none"] for k in results if k != "none"}
    return {"loss": results, "delta_vs_full": deltas}


@torch.no_grad()
def residual_rate(model: FaCT, batches: list, device: str) -> dict | None:
    """Information budget of the continuous channel, nats per token frame.

    Capped channel: the exact AWGN capacity bound. KL channel: the mean
    per-frame KL to the N(0, I) prior over valid frames, an upper bound on
    the information it carries (the VAE "rate", Alemi et al. 2018).
    Discrete streams are reported next to it as ln(perplexity), the
    unigram entropy per frame.
    """
    bn = model.bottleneck
    if bn.residual_down is None:
        return None
    frame_rate = model.cfg.audio.sample_rate / (
        model.cfg.audio.hop_length * model.cfg.encoder.frame_stack)
    if bn.residual_capacity > 0:
        nats, kind = bn.residual_capacity, "capacity_bound"
    else:
        s = model.cfg.encoder.frame_stack
        tot = n = 0.0
        for batch in batches:
            out = model.encode_mel(batch.mel.to(device))
            t_tok = out.kl_loss.shape[1]
            lengths = token_lengths(batch, t_tok, s)
            for i, k in enumerate(lengths.tolist()):
                # kl_loss is the per-dim mean; the rate sums over dims.
                tot += float(out.kl_loss[i, :k].sum()) * bn.residual_dim
                n += k
        nats, kind = tot / max(1.0, n), "kl_upper_bound"
    return {"kind": kind, "nats_per_frame": nats,
            "bits_per_second": nats / math.log(2) * frame_rate}


@torch.no_grad()
def _f0_features(model: FaCT, batches: list, device: str) -> tuple[dict, torch.Tensor]:
    feats: dict[str, list] = {"content": [], "prosody": []}
    targets = []
    for batch in batches:
        out = model.encode_mel(batch.mel.to(device))
        t_tok = out.primary_emb().shape[1]
        f0 = batch.f0_bins.to(device)[:, :t_tok]
        valid = f0 >= 0  # padding is -100
        if out.content_emb is not None:
            feats["content"].append(out.content_emb[valid].cpu())
        if out.prosody_emb is not None:
            feats["prosody"].append(out.prosody_emb[valid].cpu())
        targets.append(f0[valid].cpu())
    return ({k: torch.cat(v) for k, v in feats.items() if v}, torch.cat(targets))


def quick_f0_probes(model: FaCT, batches: list, device: str) -> dict:
    """content->F0 vs prosody->F0 accuracy on frozen features.

    Probes train on the first 80% of utterances and are scored on the
    held-out 20%, so accuracy measures recoverable F0, not memorization.
    `f0_majority_acc` is the always-predict-the-commonest-bin baseline.
    """
    n_eval = max(1, len(batches) // 5)
    if len(batches) < 2:
        raise SystemExit("tripwire probes need >= 2 batches (raise --n-utts)")
    tr_x, tr_y = _f0_features(model, batches[:-n_eval], device)
    ev_x, ev_y = _f0_features(model, batches[-n_eval:], device)
    n_classes = model.cfg.prosody.f0_bins + 1
    majority = int(torch.bincount(tr_y, minlength=n_classes).argmax())
    report = {"f0_majority_acc": float((ev_y == majority).float().mean()),
              "train_frames": int(tr_y.numel()), "eval_frames": int(ev_y.numel())}

    def chunks(x, y):
        def gen():
            for i in range(0, x.shape[0], 2048):
                yield x[i : i + 2048].unsqueeze(0), y[i : i + 2048].unsqueeze(0)
        return gen

    for name in tr_x:
        report[f"{name}_to_f0_acc"] = probe_accuracy(
            chunks(tr_x[name], tr_y), in_dim=tr_x[name].shape[-1],
            n_classes=n_classes, steps=300, eval_batches=chunks(ev_x[name], ev_y),
        )
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-utts", type=int, default=256,
                    help="content codebook is 8192 codes: utilization needs >> 8192 tokens")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    # A sweep arm's checkpoint differs from --config in the swept field;
    # the checkpoint's own config is the one it was trained with.
    if "config" in ckpt:
        cfg = FaCTConfig.from_dict(ckpt["config"])
    else:
        cfg = FaCTConfig.from_yaml(args.config)
    model = FaCT(cfg)
    model.load_state_dict(ckpt["model"])
    model = model.to(args.device).eval()

    ds = ShardedSpeech(args.shards, cfg)
    idxs = list(range(0, len(ds), max(1, len(ds) // args.n_utts)))[: args.n_utts]
    batches = [collate([ds[i] for i in idxs[j : j + 4]], cfg)
               for j in range(0, len(idxs), 4)]

    report: dict = {"ckpt": args.ckpt, "step": ckpt["step"]}

    s = cfg.encoder.frame_stack
    content_idx, prosody_idx = [], []
    for b in batches:
        tok = model.tokenize(mel=b.mel.to(args.device))
        if tok.content_indices is None:
            continue
        lengths = token_lengths(b, tok.content_indices.shape[1], s)
        content_idx.append(valid_indices(tok.content_indices, lengths))
        if tok.prosody_indices is not None:
            prosody_idx.append(valid_indices(
                tok.prosody_indices, lengths, model.bottleneck.prosody_rate_divisor))
    if content_idx:
        report["content"] = codebook_stats(
            torch.cat(content_idx), model.bottleneck.content_codebook_size)
    if prosody_idx:
        report["prosody"] = codebook_stats(
            torch.cat(prosody_idx), model.bottleneck.prosody_codebook_size)
    report["greedy_byte_cer"] = greedy_byte_cer(model, batches, args.device)
    torch.manual_seed(0)  # capped residual channels sample noise
    report["residual_rate"] = residual_rate(model, batches, args.device)
    if report["residual_rate"] is not None:
        rr = report["residual_rate"]
        rr["discrete_nats_per_frame"] = sum(
            math.log(report[k]["perplexity"]) / (
                model.bottleneck.prosody_rate_divisor if k == "prosody" else 1)
            for k in ("content", "prosody") if k in report)
    report["channel_zeroing"] = channel_zeroing_deltas(model, batches, args.device)
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
    # R3: no grounding if greedy-CTC CER > 40% at the end of the pilot.
    flags["r3_ctc_ungrounded"] = report["greedy_byte_cer"] > 0.40
    report["flags"] = flags

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
    if any(flags.values()):
        print("\nWARN: tripwire(s) fired - see RISK_ASSESSMENT SS2 for pivots.")


if __name__ == "__main__":
    main()
