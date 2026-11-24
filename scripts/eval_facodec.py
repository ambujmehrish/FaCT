"""FACodec head-to-head with FaCT: the factorization tests of
scripts/eval_factorization.py and the voice-conversion tests of
scripts/eval_vc.py, run on FACodec's own streams (fact/baselines/facodec.py),
with the same metric names so the two models line up row for row.

    python scripts/eval_facodec.py --eval-dir <eval>/librispeech_test_clean_full \
        --out results/facodec/ls_shard0.json --shard 0 --num-shards 8 \
        --wer-model large-v3 --utmos

Merge: python -m fact.eval.factorization merge --inputs 'results/facodec/ls_shard*.json' --out results/facodec/ls.json

Streams swapped exactly as for FaCT: prosody codes (1), content codes (2),
acoustic-detail codes (3), timbre vector. Views: "full" = all codes,
"discrete_only" = without the acoustic-detail codes (FACodec's analogue of
dropping FaCT's residual; named like FaCT's view so tables align).
Speaker: FACodec's timbre vector of the utterance itself for the
factorization conditions (its native protocol, no enrolment), the target's
for voice conversion.

Factorization conditions (WORLD variants w0 / s+K / s-K / flat, all
resynthesized so WORLD artifacts cancel; codes swapped between variants):
  recon, prosody<s>, content<s>, both<s>, prosody_flat; token agreement of the
  content and prosody codes between w0 and each variant.
Voice conversion (see fact/eval/vc.py): vc_same / vc_cross / vcpm_same /
vcpm_cross with held-out target similarity, pitch gap, intonation correlation.
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
from fact.eval import vc  # noqa: E402
from fact.eval.eval_set import (load_eval_manifest, load_eval_wavs, load_transcripts,  # noqa: E402
                                reference_partners, speaker_key)
from fact.eval.metrics_audio import resample  # noqa: E402

SR = 16_000
HOP = 160  # F0 frame for the metrics (10 ms); pitch statistics are ratios, so any hop works
F0_MIN, F0_MAX = 50.0, 550.0
VIEWS = ("discrete_only", "full")


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--eval-dir", type=Path)
    src.add_argument("--eval-manifest", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shifts", default="4,-4", help="semitones, comma-separated")
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--wer-model", default=None)
    ap.add_argument("--utmos", action="store_true")
    ap.add_argument("--no-vc", action="store_true", help="factorization conditions only")
    ap.add_argument("--group-threshold-hz", type=float, default=160.0)
    ap.add_argument("--f0-utts-per-speaker", type=int, default=3)
    ap.add_argument("--max-shift-st", type=float, default=12.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    if not 0 <= a.shard < a.num_shards:
        ap.error(f"--shard must be in [0, {a.num_shards})")
    try:
        a.shifts = [int(s) for s in a.shifts.split(",") if s.strip()]
    except ValueError:
        ap.error("--shifts: comma-separated integers")
    if not a.shifts or 0 in a.shifts:
        ap.error("--shifts: non-empty, no 0")
    a.views = [v for v in a.views.split(",") if v]
    if not a.views or set(a.views) - set(VIEWS):
        ap.error("--views: discrete_only and/or full")
    if not 50.0 < a.group_threshold_hz < 400.0:
        ap.error("--group-threshold-hz: between 50 and 400")
    src_path = a.eval_dir or a.eval_manifest
    if not src_path.exists():
        ap.error(f"{src_path} not found")
    return a


def main(argv=None, codec=None) -> None:
    """`codec` is for tests (a stand-in with FACodecBaseline's interface)."""
    a = parse_args(argv)
    dev = a.device
    if codec is None:
        from fact.baselines.facodec import FACodecBaseline
        codec = FACodecBaseline(view="full", device=dev)
    sl = codec.stream_slices()

    if a.eval_dir:
        items = load_eval_wavs(a.eval_dir, a.limit)
        transcripts = load_transcripts(eval_dir=a.eval_dir)
    else:
        items = load_eval_manifest(a.eval_manifest, a.limit, SR)
        transcripts = load_transcripts(manifest=a.eval_manifest)
    wavs = {k: (resample(w.float(), r, SR) if r != SR else w.float()) for k, w, r in items}
    keys = sorted(wavs)
    partner = reference_partners(keys, skip_singletons=True)
    by_spk: dict[str, list[str]] = {}
    for k in partner:
        by_spk.setdefault(speaker_key(k), []).append(k)
    by_spk = {sp: sorted(v) for sp, v in by_spk.items()}

    def f0_of(w: torch.Tensor) -> np.ndarray:
        return extract_f0_pyworld(w.detach().cpu().float().numpy(), SR, HOP, F0_MIN, F0_MAX)

    spk_f0 = {sp: vc.speaker_median_f0([f0_of(wavs[k]) for k in v[: a.f0_utts_per_speaker]])
              for sp, v in sorted(by_spk.items())}
    groups = vc.pitch_groups(spk_f0, a.group_threshold_hz)
    targets = vc.choose_targets(groups) if not a.no_vc else {}
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

    def mix(base_ids, **repl):
        """Codes of `base_ids` with whole streams replaced, e.g. prosody=ids_s."""
        ids = base_ids.clone()
        for stream, src in repl.items():
            ids[sl[stream]] = src[sl[stream]]
        return ids

    def fit(w, n):
        w = w[0].float().cpu()
        return w[:n] if len(w) >= n else torch.nn.functional.pad(w, (0, n - len(w)))

    per, skipped = {}, {}
    t0 = time.time()
    for i, k in enumerate(own):
        src = wavs[k]
        a_spk = speaker_key(k)
        n = len(src)
        try:
            var = {nm: torch.from_numpy(v) for nm, v in
                   fz.world_variants(src.numpy(), SR, a.shifts, F0_MIN, F0_MAX).items()}
            tg = {p: t for p, t in targets.get(a_spk, {}).items() if t is not None}
            exp = {p: vc.expected_shift_st(spk_f0[t], spk_f0[a_spk], a.max_shift_st) for p, t in tg.items()}
            shifted = {p: torch.from_numpy(vc.world_shift(src.numpy(), SR, exp[p], F0_MIN, F0_MAX))
                       for p in tg}
        except ValueError as e:
            skipped[k] = str(e)
            continue
        enc = {nm: codec.encode(w[None]) for nm, w in var.items()}
        T = min(e["ids"].shape[-1] for e in enc.values())
        ids = {nm: e["ids"][..., :T] for nm, e in enc.items()}
        spk_own = enc["w0"]["spk"]
        m: dict[str, float] = {}
        for nm in var:
            if nm != "w0":
                m[f"tokens/content_agree/{nm}"] = fz.token_agreement(ids["w0"][sl["content"]], ids[nm][sl["content"]])
                m[f"tokens/prosody_agree/{nm}"] = fz.token_agreement(ids["w0"][sl["prosody"]], ids[nm][sl["prosody"]])
        conds = {"recon": (ids["w0"], spk_own, None)}
        for s in a.shifts:
            nm = f"s{s:+d}"
            conds[f"prosody{s:+d}"] = (mix(ids["w0"], prosody=ids[nm]), spk_own, None)
            conds[f"content{s:+d}"] = (mix(ids["w0"], content=ids[nm]), spk_own, None)
            conds[f"both{s:+d}"] = (mix(ids["w0"], content=ids[nm], prosody=ids[nm]), spk_own, None)
        conds["prosody_flat"] = (mix(ids["w0"], prosody=ids["flat"]), spk_own, None)
        # Voice conversion on the original (non-WORLD) audio, as for FaCT.
        if tg:
            src_enc = codec.encode(src[None])
            for p, t in tg.items():
                ref, held = by_spk[t][0], by_spk[t][1]
                spk_t = codec.encode(wavs[ref][None])["spk"]
                sh = codec.encode(shifted[p][None])["ids"]
                Tv = min(sh.shape[-1], src_enc["ids"].shape[-1])
                conds[f"vc_{p}"] = (src_enc["ids"][..., :Tv], spk_t, t)
                conds[f"vcpm_{p}"] = (mix(src_enc["ids"][..., :Tv], prosody=sh[..., :Tv]), spk_t, t)
                m[f"pair/{p}/expected_st"] = exp[p]
                m[f"pair/{p}/expected_abs_st"] = abs(exp[p])
                m[f"real/{p}/sim_same_spk"] = sim(wavs[ref], wavs[held], SR)
                m[f"real/{p}/sim_cross"] = sim(src, wavs[held], SR)
        f0_src = f0_of(var["w0"])
        f0_orig = f0_of(src)
        for view in a.views:
            use_res = view == "full"
            outs = {c: fit(codec.decode(cid, spk, use_residual=use_res), n) for c, (cid, spk, _) in conds.items()}
            f0s = {c: f0_of(w) for c, w in outs.items()}
            base_f0 = f0s["recon"]
            m[f"{view}/recon/shift_vs_input_st"] = fz.pitch_shift_st(base_f0, f0_src)
            m[f"{view}/recon/intonation_ratio"] = fz.intonation_ratio(base_f0, f0_src)
            m[f"{view}/prosody_flat/intonation_ratio"] = fz.intonation_ratio(f0s["prosody_flat"], base_f0)
            m[f"{view}/recon/sim_source"] = sim(src, outs["recon"], SR)
            for c, (_, _, t) in conds.items():
                key = f"{view}/{c}"
                if c != "recon" and not c.startswith("vc"):
                    m[f"{key}/shift_st"] = fz.pitch_shift_st(f0s[c], base_f0)
                if t is not None:
                    out = outs[c]
                    m[f"{key}/sim_src"] = sim(src, out, SR)
                    m[f"{key}/sim_tgt"] = sim(wavs[by_spk[t][1]], out, SR)
                    m[f"{key}/tgt_wins"] = float(m[f"{key}/sim_tgt"] > m[f"{key}/sim_src"])
                    m[f"{key}/pitch_gap_st"] = 12.0 * np.log2(fz.median_f0(f0s[c]) / spk_f0[t])
                    m[f"{key}/pitch_gap_abs_st"] = abs(m[f"{key}/pitch_gap_st"])
                    m[f"{key}/contour_corr"] = vc.contour_corr(f0s[c], f0_orig)
                if utmos is not None and (c == "recon" or t is not None):
                    m[f"{key}/utmos"] = utmos(outs[c], SR)
                if wer is not None:
                    e, nw = wer(outs[c], SR, transcripts[k])
                    m[f"{key}/wer_errors"], m[f"{key}/wer_words"] = e, nw
        per[k] = {kk: (None if isinstance(v, float) and not np.isfinite(v) else float(v))
                  for kk, v in m.items()}
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(own)} utterances ({time.time() - t0:.0f}s)", flush=True)

    res = {"codec": codec.name, "discrete_bps_full": float(codec.discrete_bps),
           "frame_rate_hz": float(codec.frame_rate_hz), "shifts": a.shifts, "views": a.views,
           "view_meaning": {"full": "all codes", "discrete_only": "no acoustic-detail codes"},
           "eval": str(a.eval_dir or a.eval_manifest), "num_shards": a.num_shards,
           "wer_model": a.wer_model, "utmos": a.utmos, "vc": not a.no_vc,
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
