"""Voice-conversion evaluation of a FaCT stage-D decoder (fact/eval/vc.py
has the conditions). One shard per job:

    python scripts/eval_vc.py --decoder <runs>/wavedec_main_c8_rd05_la4/step_0250000.pt \
        --eval-dir <eval>/librispeech_test_clean_full --out results/vc/la4_ls_shard0.json \
        --shard 0 --num-shards 4 --wer-model large-v3 --utmos

Merge the shards with
    python -m fact.eval.factorization merge --inputs 'results/vc/la4_ls_shard*.json' --out results/vc/la4_ls.json

Per source utterance and view (full, discrete_only), conditions recon,
vc_same, vc_cross, vcpm_same, vcpm_cross; metrics per condition:
  wer_errors/wer_words  Whisper vs the ground-truth transcript
  utmos                 naturalness
  sim_src               WavLM-SV cosine to the source utterance
  sim_tgt               to a HELD-OUT utterance of the target (not the one
                        the speaker vector was computed from)
  tgt_wins              1 if sim_tgt > sim_src
  pitch_gap_st          output median F0 vs the target speaker's, semitones
                        (+ pitch_gap_abs_st: low->high and high->low signs cancel in a mean)
  contour_corr          log-F0 correlation with the source (intonation kept)
  rtf                   model compute (encoder + decoder) / audio duration
and per pair: expected_st (median-F0 ratio), real/<pair>/sim_same_spk (two
real utterances of the target: the ceiling) and real/<pair>/sim_cross
(source vs target, real speech: the floor).
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
from fact.eval import vc  # noqa: E402
from fact.eval.eval_set import (load_eval_manifest, load_eval_wavs, load_transcripts,  # noqa: E402
                                reference_partners, speaker_key)
from fact.eval.factorization import median_f0, pitch_shift_st  # noqa: E402
from fact.eval.metrics_audio import resample  # noqa: E402

VIEWS = ("discrete_only", "full")


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", required=True, help="stage-D checkpoint (step_*.pt)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--eval-dir", type=Path)
    src.add_argument("--eval-manifest", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--wer-model", default=None)
    ap.add_argument("--utmos", action="store_true")
    ap.add_argument("--group-threshold-hz", type=float, default=160.0,
                    help="speaker median F0 below this = 'low' pitch group")
    ap.add_argument("--f0-utts-per-speaker", type=int, default=3)
    ap.add_argument("--max-shift-st", type=float, default=12.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    # Validate everything that can fail BEFORE loading models or audio.
    if not 0 <= a.shard < a.num_shards:
        ap.error(f"--shard must be in [0, {a.num_shards})")
    a.views = [v for v in a.views.split(",") if v]
    if not a.views or set(a.views) - set(VIEWS):
        ap.error("--views: discrete_only and/or full")
    if not 50.0 < a.group_threshold_hz < 400.0:
        ap.error("--group-threshold-hz: between 50 and 400")
    if a.f0_utts_per_speaker < 1 or a.max_shift_st <= 0:
        ap.error("--f0-utts-per-speaker >= 1 and --max-shift-st > 0")
    if not Path(a.decoder).is_file():
        ap.error(f"--decoder {a.decoder} not found")
    src_path = a.eval_dir or a.eval_manifest
    if not src_path.exists():
        ap.error(f"{src_path} not found")
    return a


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
    partner = reference_partners(keys, skip_singletons=True)  # speakers with >= 2 utterances
    by_spk: dict[str, list[str]] = {}
    for k in partner:
        by_spk.setdefault(speaker_key(k), []).append(k)
    by_spk = {sp: sorted(v) for sp, v in by_spk.items()}

    def f0_of(w: torch.Tensor) -> np.ndarray:
        return extract_f0_pyworld(w.detach().cpu().numpy(), sr, hop, pcfg.f0_min, pcfg.f0_max)

    # Speaker pitch table from the first few utterances of each speaker
    # (deterministic, identical in every shard).
    spk_f0 = {sp: vc.speaker_median_f0([f0_of(wavs[k]) for k in v[: a.f0_utts_per_speaker]])
              for sp, v in sorted(by_spk.items())}
    groups = vc.pitch_groups(spk_f0, a.group_threshold_hz)
    targets = vc.choose_targets(groups)
    n_low = sum(g == "low" for g in groups.values())
    print(f"{len(groups)} speakers with a pitch: {n_low} low, {len(groups) - n_low} high "
          f"(threshold {a.group_threshold_hz:.0f} Hz)", flush=True)
    own = [k for k in keys if k in partner and speaker_key(k) in groups][a.shard::a.num_shards]

    wer = None
    if a.wer_model:
        if not transcripts:
            raise SystemExit("--wer-model needs transcripts")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from eval_factorization import build_wer  # noqa: E402
        wer = build_wer(a.wer_model, dev)
    from fact.eval.quality import SpeakerSim
    sim = SpeakerSim(dev)
    utmos = None
    if a.utmos:
        from fact.eval.quality import UTMOS
        utmos = UTMOS(dev)

    def mel_of(w: torch.Tensor) -> torch.Tensor:
        m = mel_fn(w.to(dev)[None])
        return m[:, : m.shape[1] // s * s]

    def sync():
        if dev.startswith("cuda"):
            torch.cuda.synchronize()

    @torch.no_grad()
    def decode(c_out, p_out, spk, view, n_frames):
        cond = c_out.content_emb + p_out.prosody_emb
        if view == "full" and c_out.residual_emb is not None:
            cond = cond + c_out.residual_emb
        cond = cond.repeat_interleave(s, dim=1)[:, :n_frames]
        x = torch.cat([cond, spk[:, None].expand(-1, cond.shape[1], -1)], dim=-1)
        return gen(x)[0].float().cpu()

    @torch.no_grad()
    def spk_vec(w: torch.Tensor) -> torch.Tensor:
        return fact.speaker_encoder(fact._norm_mel(mel_of(w)))

    per, skipped = {}, {}
    t0 = time.time()
    for i, k in enumerate(own):
        src = wavs[k]
        a_spk = speaker_key(k)
        tg = {p: t for p, t in targets[a_spk].items() if t is not None}
        try:
            shifted = {p: torch.from_numpy(vc.world_shift(
                src.numpy(), sr, vc.expected_shift_st(spk_f0[t], spk_f0[a_spk], a.max_shift_st),
                pcfg.f0_min, pcfg.f0_max)) for p, t in tg.items()}
        except ValueError as e:
            skipped[k] = str(e)
            continue
        mel_src = mel_of(src)
        n_frames = mel_src.shape[1]
        with torch.no_grad():
            enc_src = fact.encode_mel(mel_src)
            enc_sh = {p: fact.encode_mel(mel_of(w)[:, :n_frames]) for p, w in shifted.items()}
        f0_src = f0_of(src)
        m: dict[str, float] = {}
        conds = {"recon": (enc_src, spk_vec(wavs[partner[k]]), None)}
        for p, t in tg.items():
            ref, held = by_spk[t][0], by_spk[t][1]
            v_t = spk_vec(wavs[ref])
            conds[f"vc_{p}"] = (enc_src, v_t, t)
            conds[f"vcpm_{p}"] = (enc_sh[p], v_t, t)
            m[f"pair/{p}/expected_st"] = vc.expected_shift_st(spk_f0[t], spk_f0[a_spk], a.max_shift_st)
            m[f"pair/{p}/expected_abs_st"] = abs(m[f"pair/{p}/expected_st"])  # signs cancel in a mean
            m[f"real/{p}/sim_same_spk"] = sim(wavs[ref], wavs[held], sr)
            m[f"real/{p}/sim_cross"] = sim(src, wavs[held], sr)
        for view in a.views:
            for c, (p_enc, spk, t) in conds.items():
                sync()
                t1 = time.time()
                out = decode(enc_src, p_enc, spk, view, n_frames)
                sync()
                dt = time.time() - t1
                key = f"{view}/{c}"
                f_out = f0_of(out)
                m[f"{key}/sim_src"] = sim(src, out, sr)
                m[f"{key}/contour_corr"] = vc.contour_corr(f_out, f0_src)
                m[f"{key}/shift_vs_src_st"] = pitch_shift_st(f_out, f0_src)
                if c == "vc_same" or c == "recon":
                    # decoder time + one encoder pass, per second of audio (vcpm adds a second pass)
                    m[f"{key}/rtf"] = dt / (len(src) / sr)
                if t is not None:
                    held = wavs[by_spk[t][1]]
                    m[f"{key}/sim_tgt"] = sim(held, out, sr)
                    m[f"{key}/tgt_wins"] = float(m[f"{key}/sim_tgt"] > m[f"{key}/sim_src"])
                    m[f"{key}/pitch_gap_st"] = 12.0 * np.log2(median_f0(f_out) / spk_f0[t])
                    m[f"{key}/pitch_gap_abs_st"] = abs(m[f"{key}/pitch_gap_st"])
                if utmos is not None:
                    m[f"{key}/utmos"] = utmos(out, sr)
                if wer is not None:
                    e, n = wer(out, sr, transcripts[k])
                    m[f"{key}/wer_errors"], m[f"{key}/wer_words"] = e, n
        sync()
        t1 = time.time()
        with torch.no_grad():
            fact.encode_mel(mel_src)
        sync()
        enc_rtf = (time.time() - t1) / (len(src) / sr)
        for view in a.views:
            for c in ("recon", "vc_same"):
                if f"{view}/{c}/rtf" in m:
                    m[f"{view}/{c}/rtf"] += enc_rtf
        per[k] = {kk: (None if isinstance(v, float) and not np.isfinite(v) else float(v))
                  for kk, v in m.items()}
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(own)} utterances ({time.time() - t0:.0f}s)", flush=True)

    res = {"decoder": str(a.decoder), "fact_ckpt": meta["fact_ckpt"], "views": a.views,
           "eval": str(a.eval_dir or a.eval_manifest), "num_shards": a.num_shards,
           "wer_model": a.wer_model, "utmos": a.utmos,
           "group_threshold_hz": a.group_threshold_hz, "max_shift_st": a.max_shift_st,
           "speaker_f0": {sp: (f if np.isfinite(f) else None) for sp, f in spk_f0.items()},
           "speaker_group": groups, "targets": targets,
           "shard": a.shard, "skipped": skipped, "per_utterance": per}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(f"{a.out.name}.{os.getpid()}.partial")
    from fact.provenance import provenance
    res["provenance"] = provenance()
    tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    os.replace(tmp, a.out)
    print(f"wrote {a.out}: {len(per)} utterances, {len(skipped)} skipped")


if __name__ == "__main__":
    main()
