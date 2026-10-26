"""Zero-shot TTS evaluation of a trained TTS LM (fact.tts.train).

    python scripts/eval_tts.py --lm <runs>/tts_fact/step_0200000.pt \
        --wave-decoder <runs>/wavedec_main_c8_rd05/step_0250000.pt \
        --eval-dir <eval>/librispeech_test_clean_full --out results/tts/fact_ls_shard0.json \
        --shard 0 --num-shards 8 --temperature 1.0 --top-k 50 --wer-model large-v3

Protocol (fixed before any result; identical for every codec):
- Targets: utterances of 4-10 s (LibriSpeech test-clean: the standard
  zero-shot subset), or the first --limit-targets of a manifest (Emilia dev,
  for choosing sampling settings - never the test sets).
- Prompt (cross-sentence): another utterance of the SAME speaker - the
  shortest one of at least 3 s (ties: sorted key); targets with none are
  skipped and listed. The LM continues the prompt's tokens; its text input
  is prompt transcript + " " + target transcript.
- Text: LibriSpeech transcripts are upper-case without punctuation while the
  LMs were trained on Emilia's cased, punctuated text, so every transcript is
  put in sentence case (lower-case, first letter capital, final period) -
  the same for every codec; WER is scored against the original transcript
  with Whisper's English normalizer. Manifest text is used as is.
- Sampling: one seed per utterance (stable hash of its key), so all codecs
  draw the same random stream per utterance; settings via flags.
- Metrics: corpus WER (Whisper), speaker similarity of the output to the
  prompt audio, UTMOS, output duration and its ratio to the ground truth.
  A generation with < 2 frames counts as all words deleted.
Shards: utterances[shard::num_shards]; merge with
    python -m fact.eval.factorization merge --inputs 'results/tts/x_shard*.json' --out ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fact.eval.eval_set import (load_eval_manifest, load_eval_wavs, load_transcripts,  # noqa: E402
                                speaker_key)
from fact.eval.metrics_audio import resample  # noqa: E402


def sentence_case(text: str) -> str:
    t = " ".join(text.strip().split()).lower()
    if not t:
        return t
    t = t[0].upper() + t[1:]
    return t if t[-1] in ".!?" else t + "."


def text_ids(text: str) -> torch.Tensor:
    return (torch.tensor(list(text.encode("utf-8")), dtype=torch.long) + 1).clamp(1, 256)


def choose_prompts(durs: dict[str, float], targets: list[str], min_prompt: float = 3.0) -> dict:
    by_spk: dict[str, list[str]] = {}
    for k in sorted(durs):
        by_spk.setdefault(speaker_key(k), []).append(k)
    out = {}
    for k in targets:
        cands = [c for c in by_spk[speaker_key(k)] if c != k and durs[c] >= min_prompt]
        if cands:
            out[k] = min(cands, key=lambda c: (durs[c], c))
    return out


def seed_of(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", required=True)
    ap.add_argument("--wave-decoder", default=None, help="stage-D checkpoint (FaCT LMs)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--eval-dir", type=Path)
    src.add_argument("--eval-manifest", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--min-target", type=float, default=4.0)
    ap.add_argument("--max-target", type=float, default=10.0)
    ap.add_argument("--min-prompt", type=float, default=3.0)
    ap.add_argument("--limit-targets", type=int, default=None, help="first N targets (dev selection)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--wer-model", default=None)
    ap.add_argument("--no-quality", action="store_true", help="skip UTMOS and speaker similarity")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    if not 0 <= a.shard < a.num_shards:
        ap.error(f"--shard must be in [0, {a.num_shards})")
    if not Path(a.lm).is_file():
        ap.error(f"--lm {a.lm} not found")
    if not 0 < a.min_target <= a.max_target:
        ap.error("need 0 < --min-target <= --max-target")
    if a.temperature < 0 or a.top_k < 0 or not 0 < a.top_p <= 1:
        ap.error("bad sampling settings")
    src_path = a.eval_dir or a.eval_manifest
    if not src_path.exists():
        ap.error(f"{src_path} not found")
    return a


def main(argv=None) -> None:
    a = parse_args(argv)
    from fact.tts.codecs import build_codec
    from fact.tts.model import TTSLM, TTSConfig

    dev = a.device
    ck = torch.load(a.lm, map_location="cpu", weights_only=True)
    cfg = TTSConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["config"].items()})
    lm = TTSLM(cfg)
    lm.load_state_dict(ck["model"])
    lm = lm.to(dev).eval()
    codec_meta = ck["run"]["codec_meta"]
    codec = build_codec(codec_meta, dev, a.wave_decoder)
    sr = codec.sample_rate

    if a.eval_dir:
        items = load_eval_wavs(a.eval_dir, None)
        transcripts = load_transcripts(eval_dir=a.eval_dir)
        norm_text = sentence_case
    else:
        items = load_eval_manifest(a.eval_manifest, None, sr)
        transcripts = load_transcripts(manifest=a.eval_manifest)
        norm_text = lambda t: " ".join(t.strip().split())  # noqa: E731
    wavs = {k: (resample(w.float(), r, sr) if r != sr else w.float()) for k, w, r in items}
    durs = {k: w.shape[0] / sr for k, w in wavs.items()}
    missing = [k for k in wavs if k not in transcripts]
    if missing:
        raise SystemExit(f"{len(missing)} utterances without transcripts (e.g. {missing[:2]})")
    targets = sorted(k for k in wavs if a.min_target <= durs[k] <= a.max_target)
    if a.limit_targets:
        targets = targets[: a.limit_targets]
    prompts = choose_prompts(durs, targets, a.min_prompt)
    skipped = {k: "no same-speaker prompt >= min-prompt" for k in targets if k not in prompts}
    own = [k for k in targets if k in prompts][a.shard::a.num_shards]

    wer = utmos = sim = None
    if a.wer_model:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from eval_factorization import build_wer
        wer = build_wer(a.wer_model, dev)
    if not a.no_quality:
        from fact.eval.quality import UTMOS, SpeakerSim
        utmos, sim = UTMOS(dev), SpeakerSim(dev)

    per = {}
    t0 = time.time()
    for i, k in enumerate(own):
        p = prompts[k]
        txt = text_ids(norm_text(transcripts[p]) + " " + norm_text(transcripts[k]))
        if txt.numel() > cfg.max_text:
            skipped[k] = f"text {txt.numel()} bytes > max_text {cfg.max_text}"
            continue
        pc = codec.encode(wavs[p])
        room = cfg.max_frames - pc.shape[1]
        if room < 2:
            skipped[k] = "prompt leaves no room for generation"
            continue
        g = torch.Generator().manual_seed(seed_of(k))
        new = lm.generate(txt, pc, max_new_frames=room, temperature=a.temperature,
                          top_k=a.top_k, top_p=a.top_p, generator=g)
        m = {"frames": float(new.shape[1]), "prompt_s": durs[p], "target_s": durs[k]}
        if new.shape[1] < 2:
            m["failed"] = 1.0
            if wer is not None:
                m["wer_errors"] = float(len(transcripts[k].split()))
                m["wer_words"] = m["wer_errors"]
        else:
            out = codec.decode(new, wavs[p])
            m.update(failed=0.0, dur_s=out.shape[0] / sr, dur_ratio=out.shape[0] / sr / durs[k])
            if wer is not None:
                m["wer_errors"], m["wer_words"] = wer(out, sr, transcripts[k])
            if utmos is not None:
                m["utmos"] = utmos(out, sr)
                m["sim_prompt"] = sim(wavs[p], out, sr)
        per[k] = {f"tts/{n}": v for n, v in m.items()}  # "/" names: merge aggregates them
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(own)} ({time.time() - t0:.0f}s)", flush=True)

    res = {"lm": str(a.lm), "codec": codec_meta["codec"], "codec_meta": codec_meta,
           "wave_decoder": a.wave_decoder, "eval": str(a.eval_dir or a.eval_manifest),
           "num_shards": a.num_shards, "shard": a.shard,
           "sampling": {"temperature": a.temperature, "top_k": a.top_k, "top_p": a.top_p},
           "targets": {"min_s": a.min_target, "max_s": a.max_target, "limit": a.limit_targets,
                       "min_prompt_s": a.min_prompt, "n_total": len(targets)},
           "wer_model": a.wer_model, "skipped": skipped, "per_utterance": per}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(f"{a.out.name}.{os.getpid()}.partial")
    tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    os.replace(tmp, a.out)
    print(f"wrote {a.out}: {len(per)} utterances, {len(skipped)} skipped")


if __name__ == "__main__":
    main()
