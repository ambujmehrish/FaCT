"""Render codec reconstructions of a few test utterances for listening.

    python scripts/render_codec_samples.py --eval-dir <LS test-clean> --n 8 --out <dir> \
        --spec 'fact_wave:decoder=<la4_ltts step.pt>:view=full:spk_ref=other:label=FaCT la4 0.42k' \
        --spec 'focalcodec:config=lucadellalib/focalcodec_25hz:label=FocalCodec 25Hz' ...

Utterances: the first --n TTS targets of scripts/eval_tts.py (4-10 s, sorted
keys, with a same-speaker prompt of >= 3 s), so the codec, TTS and prosody
samples come from the same utterances. Specs are the eval harness's
(scripts/eval_baselines.py), so each system is set up exactly as in the
test tables. FaCT entries use the speaker vector of another utterance of the
same speaker (spk_ref=other), as in the tables.

Writes <out>/codec/<label>/<utt>.wav (and <out>/codec/original/), all at
--sample-rate and RMS-matched to --rms-dbfs for comparable loudness, plus
<out>/utterances.json (keys, transcripts, prompt keys).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_here = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _here / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def level(wav: torch.Tensor, rms_dbfs: float) -> torch.Tensor:
    """Scale to the target RMS, then limit the peak to 0.99 (never clips)."""
    w = wav.float()
    rms = w.pow(2).mean().sqrt().clamp_min(1e-8)
    w = w * (10 ** (rms_dbfs / 20) / rms)
    peak = w.abs().max()
    return w * (0.99 / peak) if peak > 0.99 else w


def pick_utterances(durs: dict[str, float], n: int, min_target=4.0, max_target=10.0, min_prompt=3.0):
    et = _load("eval_tts")
    targets = sorted(k for k in durs if min_target <= durs[k] <= max_target)[:n]
    prompts = et.choose_prompts(durs, targets, min_prompt)
    return [k for k in targets if k in prompts], prompts


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--spec", action="append", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sample-rate", type=int, default=16_000)
    ap.add_argument("--rms-dbfs", type=float, default=-26.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)

    eb, et = _load("eval_baselines"), _load("eval_tts")
    from fact.baselines import base as registry
    from fact.eval.eval_set import load_eval_wavs, load_transcripts
    from fact.eval.metrics_audio import resample

    items = load_eval_wavs(a.eval_dir, None)
    durs = {k: w.shape[0] / sr for k, w, sr in items}
    keys, prompts = pick_utterances(durs, a.n)
    if not keys:
        raise SystemExit("no utterance qualifies")
    trans = load_transcripts(eval_dir=a.eval_dir)
    by_key = {k: (w, sr) for k, w, sr in items}

    def write(label: str, k: str, wav: torch.Tensor, sr: int) -> None:
        w = resample(wav.detach().float().cpu(), sr, a.sample_rate) if sr != a.sample_rate else wav.float().cpu()
        et.save_wav(a.out / "codec" / label / f"{et.stem(k)}.wav", level(w, a.rms_dbfs), a.sample_rate)

    for k in keys:
        write("original", k, *by_key[k])
    for spec in a.spec:
        name, kwargs = eb.parse_spec(spec)
        label = str(kwargs.pop("label", eb.spec_label(spec)))
        tok = registry.build(name, device=a.device, **kwargs)
        if hasattr(tok, "set_eval_set"):  # other-utterance speaker reference, as in the tables
            tok.set_eval_set(items)
        for k in keys:
            w, sr = by_key[k]
            ref = resample(w.float(), sr, tok.sample_rate).to(a.device)
            with torch.no_grad():
                if getattr(tok, "takes_key", False):
                    hyp = tok.reconstruct(ref.unsqueeze(0), key=k)
                else:
                    hyp = tok.reconstruct(ref.unsqueeze(0))
            write(label, k, hyp.squeeze(0), tok.sample_rate)
        print(f"{label}: {len(keys)} utterances", flush=True)
        del tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    meta = {"eval_dir": str(a.eval_dir), "sample_rate": a.sample_rate, "rms_dbfs": a.rms_dbfs,
            "specs": a.spec,
            "utterances": [{"key": k, "stem": et.stem(k), "text": trans.get(k, ""),
                            "prompt": prompts[k], "prompt_stem": et.stem(prompts[k])} for k in keys]}
    a.out.mkdir(parents=True, exist_ok=True)
    from fact.provenance import provenance
    meta["provenance"] = provenance()
    (a.out / "utterances.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
