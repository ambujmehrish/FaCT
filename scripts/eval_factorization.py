"""Factorization evaluation of a FaCT model through its stage-D decoder
(see fact/eval/factorization.py for the conditions and metrics).

    python scripts/eval_factorization.py --decoder <runs>/wavedec_main_c8_rd05/step_0250000.pt \
        --eval-dir <eval>/librispeech_test_clean_full --out results/factorization/x_shard0.json \
        --shard 0 --num-shards 8 --wer-model large-v3 --sim \
        --baselines "mimi:n_streams=2" "focalcodec:config=lucadellalib/focalcodec_25hz"

Each shard handles utterances[shard::num_shards]; merge the shards with
    python -m fact.eval.factorization merge --inputs 'results/factorization/x_shard*.json' --out ...
Speaker references: another utterance of the same speaker (the main
protocol); `vc` uses the first utterance of the next speaker instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fact.data.prosody_targets import extract_f0_pyworld  # noqa: E402
from fact.eval import factorization as fz  # noqa: E402
from fact.eval.eval_set import (load_eval_manifest, load_eval_wavs, load_transcripts,  # noqa: E402
                                reference_partners, speaker_key)
from fact.eval.metrics_audio import resample  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", required=True, help="stage-D checkpoint (step_*.pt)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--eval-dir", type=Path)
    src.add_argument("--eval-manifest", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shifts", default="4,-4", help="semitones, comma-separated")
    ap.add_argument("--views", default="discrete_only,full")
    ap.add_argument("--wer-model", default=None)
    ap.add_argument("--sim", action="store_true", help="speaker similarity for recon / vc")
    ap.add_argument("--baselines", nargs="*", default=[], help="token-agreement only")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    # Validate everything that can fail BEFORE loading models or audio.
    if not 0 <= a.shard < a.num_shards:
        ap.error(f"--shard must be in [0, {a.num_shards})")
    try:
        a.shifts = [int(s) for s in a.shifts.split(",") if s.strip()]
    except ValueError:
        ap.error("--shifts: comma-separated integers")
    if not a.shifts or 0 in a.shifts:
        ap.error("--shifts: non-empty, no 0 (w0 is always included)")
    a.views = [v for v in a.views.split(",") if v]
    if not a.views or set(a.views) - {"discrete_only", "full"}:
        ap.error("--views: discrete_only and/or full")
    if not Path(a.decoder).is_file():
        ap.error(f"--decoder {a.decoder} not found")
    src_path = a.eval_dir or a.eval_manifest
    if not src_path.exists():
        ap.error(f"{src_path} not found")
    return a


def build_wer(model_size: str, device: str):
    import whisper
    from whisper.normalizers import EnglishTextNormalizer

    from fact.eval.wer import word_edit_distance
    cache = os.environ.get("FACT_CACHE")
    model = whisper.load_model(model_size, device=device,
                               download_root=str(Path(cache) / "whisper") if cache else None)
    norm = EnglishTextNormalizer()

    def wer(wav: torch.Tensor, sr: int, ref_text: str) -> tuple[float, float]:
        text = model.transcribe(resample(wav, sr, 16_000).numpy(), language="en",
                                fp16=(device != "cpu"))["text"]
        ref_w, hyp_w = norm(ref_text).split(), norm(text).split()
        return float(word_edit_distance(ref_w, hyp_w)), float(len(ref_w))
    return wer


def main(argv=None) -> None:
    a = parse_args(argv)
    from fact.modules.mel import LogMelSpectrogram
    from fact.train.stage_d_wave_decoder import load_fact_frozen, load_wave_decoder

    dev = a.device
    gen, meta = load_wave_decoder(a.decoder, dev)
    fact = load_fact_frozen(meta["fact_ckpt"], dev)
    if fact.cfg.encoder.ssl_model:
        raise SystemExit("SSL-input FaCT models are not supported here")
    bn = fact.bottleneck
    if bn.variant != "factorized" or bn.prosody_fsq is None:
        raise SystemExit("needs a factorized (content + prosody FSQ) FaCT model")
    if "discrete_only" in a.views and bn.residual_up is not None and not meta["residual_drop"]:
        raise SystemExit("this decoder was never trained without the residual (drop discrete_only)")
    sr = fact.cfg.audio.sample_rate
    hop = fact.cfg.audio.hop_length
    s = fact.cfg.encoder.frame_stack
    pcfg = fact.cfg.prosody
    mel_fn = LogMelSpectrogram(fact.cfg.audio).to(dev)

    if a.eval_dir:
        items = load_eval_wavs(a.eval_dir, a.limit)
        transcripts = load_transcripts(eval_dir=a.eval_dir)
    else:
        items = load_eval_manifest(a.eval_manifest, a.limit, sr)
        transcripts = load_transcripts(manifest=a.eval_manifest)
    wavs = {k: (resample(w.float(), r, sr) if r != sr else w.float()) for k, w, r in items}
    keys = sorted(wavs)
    partner = reference_partners(keys, skip_singletons=True)
    by_spk: dict[str, list[str]] = {}
    for k in keys:
        by_spk.setdefault(speaker_key(k), []).append(k)
    spks = sorted(by_spk)
    if len(spks) < 2:
        raise SystemExit("vc needs at least two speakers")
    vc_target = {k: by_spk[spks[(spks.index(speaker_key(k)) + 1) % len(spks)]][0] for k in keys}
    own = [k for k in keys if k in partner][a.shard::a.num_shards]

    wer = build_wer(a.wer_model, dev) if a.wer_model else None
    if wer and not transcripts:
        raise SystemExit("--wer-model needs transcripts")
    sim = None
    if a.sim:
        from fact.eval.quality import SpeakerSim
        sim = SpeakerSim(dev)
    from fact.baselines import base as registry
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_baselines import parse_spec, spec_label  # noqa: E402
    bases = {spec_label(sp): registry.build(parse_spec(sp)[0], device=dev, **{
        k: v for k, v in parse_spec(sp)[1].items() if k != "label"}) for sp in a.baselines}

    def mel_of(w: torch.Tensor) -> torch.Tensor:
        m = mel_fn(w.to(dev)[None])
        return m[:, : m.shape[1] // s * s]

    def f0_of(w: torch.Tensor) -> np.ndarray:
        return extract_f0_pyworld(w.detach().cpu().numpy(), sr, hop, pcfg.f0_min, pcfg.f0_max)

    @torch.no_grad()
    def decode(c_out, p_out, ref_mel, view, n_frames):
        cond = c_out.content_emb + p_out.prosody_emb
        if view == "full" and c_out.residual_emb is not None:
            cond = cond + c_out.residual_emb
        cond = cond.repeat_interleave(s, dim=1)[:, :n_frames]
        spk = fact.speaker_encoder(fact._norm_mel(ref_mel))
        x = torch.cat([cond, spk[:, None].expand(-1, cond.shape[1], -1)], dim=-1)
        return gen(x)[0].float().cpu()

    per, skipped = {}, {}
    t0 = time.time()
    for i, k in enumerate(own):
        src = wavs[k]
        try:
            var = fz.world_variants(src.numpy(), sr, a.shifts, pcfg.f0_min, pcfg.f0_max)
        except ValueError as e:
            skipped[k] = str(e)
            continue
        var = {n: torch.from_numpy(v) for n, v in var.items()}
        mels = {n: mel_of(v) for n, v in var.items()}
        n_frames = min(m.shape[1] for m in mels.values())
        mels = {n: m[:, :n_frames] for n, m in mels.items()}
        with torch.no_grad():
            enc = {n: fact.encode_mel(m) for n, m in mels.items()}
        ref_same = mel_of(wavs[partner[k]])
        ref_other = mel_of(wavs[vc_target[k]])
        m: dict[str, float] = {}
        for n in var:
            if n == "w0":
                continue
            m[f"tokens/content_agree/{n}"] = fz.token_agreement(enc["w0"].content_indices, enc[n].content_indices)
            m[f"tokens/prosody_agree/{n}"] = fz.token_agreement(enc["w0"].prosody_indices, enc[n].prosody_indices)
        for label, tok in bases.items():
            def toks(w):
                ww = resample(w, sr, tok.sample_rate) if tok.sample_rate != sr else w
                return tok.tokenize(ww[None].to(dev))
            t_w0 = toks(var["w0"])
            for n in var:
                if n != "w0":
                    t_n = toks(var[n])
                    m[f"tokens/{label}_agree/{n}"] = float(np.mean(
                        [fz.token_agreement(x.cpu(), y.cpu()) for x, y in zip(t_w0, t_n)]))
        conds = {"recon": ("w0", "w0", ref_same)}
        for sft in a.shifts:
            n = f"s{sft:+d}"
            conds[f"prosody{sft:+d}"] = ("w0", n, ref_same)
            conds[f"content{sft:+d}"] = (n, "w0", ref_same)
            conds[f"both{sft:+d}"] = (n, n, ref_same)
        conds["prosody_flat"] = ("w0", "flat", ref_same)
        conds["vc"] = ("w0", "w0", ref_other)
        f0_src = f0_of(var["w0"])
        for view in a.views:
            outs = {c: decode(enc[cv], enc[pv], ref, view, n_frames) for c, (cv, pv, ref) in conds.items()}
            f0s = {c: f0_of(w) for c, w in outs.items()}
            base_f0 = f0s["recon"]
            m[f"{view}/recon/shift_vs_input_st"] = fz.pitch_shift_st(base_f0, f0_src)
            for c, f in f0s.items():
                if c != "recon":
                    m[f"{view}/{c}/shift_st"] = fz.pitch_shift_st(f, base_f0)
            m[f"{view}/prosody_flat/intonation_ratio"] = fz.intonation_ratio(f0s["prosody_flat"], base_f0)
            m[f"{view}/recon/intonation_ratio"] = fz.intonation_ratio(base_f0, f0_src)
            m[f"{view}/vc/expected_st"] = 12 * np.log2(
                fz.median_f0(f0_of(wavs[vc_target[k]])) / fz.median_f0(f0_src))
            if sim is not None:
                m[f"{view}/recon/sim_source"] = sim(src, outs["recon"], sr)
                m[f"{view}/vc/sim_source"] = sim(src, outs["vc"], sr)
                m[f"{view}/vc/sim_target"] = sim(wavs[vc_target[k]], outs["vc"], sr)
            if wer is not None:
                for c, w in outs.items():
                    e, n = wer(w, sr, transcripts[k])
                    m[f"{view}/{c}/wer_errors"], m[f"{view}/{c}/wer_words"] = e, n
        per[k] = {kk: (None if isinstance(v, float) and not np.isfinite(v) else float(v))
                  for kk, v in m.items()}
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(own)} utterances ({time.time() - t0:.0f}s)", flush=True)

    res = {"decoder": str(a.decoder), "fact_ckpt": meta["fact_ckpt"], "shifts": a.shifts,
           "views": a.views, "eval": str(a.eval_dir or a.eval_manifest), "num_shards": a.num_shards,
           "baselines": sorted(bases), "wer_model": a.wer_model, "sim": a.sim,
           "shard": a.shard, "skipped": skipped, "per_utterance": per}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(f"{a.out.name}.{os.getpid()}.partial")
    tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    os.replace(tmp, a.out)
    print(f"wrote {a.out}: {len(per)} utterances, {len(skipped)} skipped")


if __name__ == "__main__":
    main()
