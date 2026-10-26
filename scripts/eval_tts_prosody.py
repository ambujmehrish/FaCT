"""Prosody control in TTS: resample stream 1 with stream 0 held fixed.

For FaCT, stream 0 = content and stream 1 = prosody, so holding content and
resampling prosody should change intonation and keep the words. Mimi-2 gets
the identical operation (codebook 0 fixed, codebook 1 resampled). Each
utterance (same targets / prompts / text handling as scripts/eval_tts.py):
  base    one sample (seed of the utterance)
  fixed   K samples with stream 0 = base's stream 0 (same length, frame-aligned)
  free    K fully free samples (reference variability)
Metrics: WER vs the ground truth per group (corpus), word agreement of each
fixed / free sample with the base sample (Whisper transcript vs Whisper
transcript), pitch diversity across {base + K samples}: s.d. of the
utterance median F0 (semitones) and of the log-F0 spread (cents); for the
frame-aligned fixed group also the mean pairwise frame-level contour
distance (cents, median over frames voiced in both); duration s.d.

    python scripts/eval_tts_prosody.py --lm <runs>/tts_fact_b1/step_0200000.pt \
        --wave-decoder <stage-D step.pt> --eval-dir <LS test-clean> \
        --out results/tts/prosody_fact_shard0.json --shard 0 --num-shards 8 \
        --resamples 3 --temperature 1.0 --top-k 50 --wer-model large-v3
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_spec = importlib.util.spec_from_file_location("eval_tts", Path(__file__).resolve().parent / "eval_tts.py")
et = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(et)

from fact.data.prosody_targets import extract_f0_pyworld  # noqa: E402
from fact.eval.eval_set import load_eval_manifest, load_eval_wavs, load_transcripts  # noqa: E402
from fact.eval.metrics_audio import resample  # noqa: E402

MIN_VOICED = 10


def median_st(f0: np.ndarray) -> float:
    v = f0[f0 > 0]
    return float(12 * np.log2(np.median(v) / 100.0)) if len(v) >= MIN_VOICED else float("nan")


def spread_cents(f0: np.ndarray) -> float:
    v = np.log2(f0[f0 > 0])
    return float(1200 * np.median(np.abs(v - np.median(v)))) if len(v) >= MIN_VOICED else float("nan")


def contour_cents(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    m = (a[:n] > 0) & (b[:n] > 0)
    if m.sum() < MIN_VOICED:
        return float("nan")
    return float(np.median(np.abs(1200 * np.log2(a[:n][m] / b[:n][m]))))


def nan_sd(x: list[float]) -> float:
    v = np.array([y for y in x if np.isfinite(y)])
    return float(np.std(v)) if len(v) >= 2 else float("nan")


def build_transcriber(model_size: str, device: str):
    import whisper
    from whisper.normalizers import EnglishTextNormalizer
    cache = os.environ.get("FACT_CACHE")
    model = whisper.load_model(model_size, device=device,
                               download_root=str(Path(cache) / "whisper") if cache else None)
    norm = EnglishTextNormalizer()

    def transcribe(wav: torch.Tensor, sr: int) -> list[str]:
        text = model.transcribe(resample(wav, sr, 16_000).numpy(), language="en",
                                fp16=(device != "cpu"))["text"]
        return norm(text).split()
    return transcribe, norm


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", required=True)
    ap.add_argument("--wave-decoder", default=None)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--eval-dir", type=Path)
    src.add_argument("--eval-manifest", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--resamples", type=int, default=3)
    ap.add_argument("--min-target", type=float, default=4.0)
    ap.add_argument("--max-target", type=float, default=10.0)
    ap.add_argument("--min-prompt", type=float, default=3.0)
    ap.add_argument("--limit-targets", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--wer-model", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    if not 0 <= a.shard < a.num_shards or a.resamples < 2:
        ap.error("need 0 <= shard < num_shards and --resamples >= 2")
    if not Path(a.lm).is_file():
        ap.error(f"--lm {a.lm} not found")

    from fact.tts.codecs import build_codec
    from fact.tts.model import TTSLM, TTSConfig
    from fact.eval.wer import word_edit_distance
    dev = a.device
    ck = torch.load(a.lm, map_location="cpu", weights_only=True)
    cfg = TTSConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["config"].items()})
    lm = TTSLM(cfg)
    lm.load_state_dict(ck["model"])
    lm = lm.to(dev).eval()
    codec_meta = ck["run"]["codec_meta"]
    codec = build_codec(codec_meta, dev, a.wave_decoder)
    sr = codec.sample_rate
    hop = sr // 50

    if a.eval_dir:
        items, transcripts, norm_text = load_eval_wavs(a.eval_dir, None), load_transcripts(eval_dir=a.eval_dir), et.sentence_case
    else:
        items = load_eval_manifest(a.eval_manifest, None, sr)
        transcripts = load_transcripts(manifest=a.eval_manifest)
        norm_text = lambda t: " ".join(t.strip().split())  # noqa: E731
    wavs = {k: (resample(w.float(), r, sr) if r != sr else w.float()) for k, w, r in items}
    durs = {k: w.shape[0] / sr for k, w in wavs.items()}
    targets = sorted(k for k in wavs if a.min_target <= durs[k] <= a.max_target)[: a.limit_targets]
    prompts = et.choose_prompts(durs, targets, a.min_prompt)
    skipped = {k: "no same-speaker prompt >= min-prompt" for k in targets if k not in prompts}
    own = [k for k in targets if k in prompts][a.shard::a.num_shards]
    transcribe = norm = None
    if a.wer_model:
        transcribe, norm = build_transcriber(a.wer_model, dev)

    def f0(w):
        return extract_f0_pyworld(w.numpy(), sr, hop, 50.0, 550.0)

    per = {}
    t0 = time.time()
    for i, k in enumerate(own):
        p = prompts[k]
        txt = et.text_ids(norm_text(transcripts[p]) + " " + norm_text(transcripts[k]))
        pc = codec.encode(wavs[p])
        room = cfg.max_frames - pc.shape[1]
        if txt.numel() > cfg.max_text or room < 2:
            skipped[k] = "text or prompt too long"
            continue
        seed = et.seed_of(k)

        def gen(s, fixed=None):
            return lm.generate(txt, pc, max_new_frames=room, temperature=a.temperature, top_k=a.top_k,
                               top_p=a.top_p, generator=torch.Generator().manual_seed(s), fixed_stream0=fixed)
        base = gen(seed)
        if base.shape[1] < 2:
            skipped[k] = "base generation < 2 frames"
            continue
        groups = {"base": [base],
                  "fixed": [gen(seed + 1000 * r, base[0]) for r in range(1, a.resamples + 1)],
                  "free": [gen(seed + 7919 * r) for r in range(1, a.resamples + 1)]}
        wav_g = {g: [codec.decode(c, wavs[p]) for c in cs] for g, cs in groups.items()}
        f0_g = {g: [f0(w) for w in ws] for g, ws in wav_g.items()}
        m: dict[str, float] = {}
        for g in ("fixed", "free"):
            tracks = f0_g["base"] + f0_g[g]
            m[f"{g}/median_st_sd"] = nan_sd([median_st(f) for f in tracks])
            m[f"{g}/spread_sd_cents"] = nan_sd([spread_cents(f) for f in tracks])
            m[f"{g}/dur_sd_s"] = float(np.std([w.shape[0] / sr for w in wav_g["base"] + wav_g[g]]))
        pairs = list(itertools.combinations(f0_g["base"] + f0_g["fixed"], 2))
        m["fixed/contour_cents"] = float(np.nanmean([contour_cents(x, y) for x, y in pairs]))
        if transcribe is not None:
            ref = norm(transcripts[k]).split()
            hyp = {g: [transcribe(w, sr) for w in ws] for g, ws in wav_g.items()}
            for g, hs in hyp.items():
                m[f"{g}/wer_errors"] = float(sum(word_edit_distance(ref, h) for h in hs))
                m[f"{g}/wer_words"] = float(len(ref) * len(hs))
            b = hyp["base"][0]
            for g in ("fixed", "free"):
                m[f"{g}_vs_base/wer_errors"] = float(sum(word_edit_distance(b, h) for h in hyp[g]))
                m[f"{g}_vs_base/wer_words"] = float(max(len(b), 1) * len(hyp[g]))
        per[k] = {n: (None if isinstance(v, float) and not np.isfinite(v) else v) for n, v in m.items()}
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(own)} ({time.time() - t0:.0f}s)", flush=True)

    res = {"lm": str(a.lm), "codec": codec_meta["codec"], "wave_decoder": a.wave_decoder,
           "eval": str(a.eval_dir or a.eval_manifest), "num_shards": a.num_shards, "shard": a.shard,
           "resamples": a.resamples,
           "sampling": {"temperature": a.temperature, "top_k": a.top_k, "top_p": a.top_p},
           "wer_model": a.wer_model, "skipped": skipped, "per_utterance": per}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(f"{a.out.name}.{os.getpid()}.partial")
    tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    os.replace(tmp, a.out)
    print(f"wrote {a.out}: {len(per)} utterances, {len(skipped)} skipped")


if __name__ == "__main__":
    main()
