"""R5 vocoder-ceiling measurement - RUN BEFORE ANY STAGE-A JOB.

Our decoder emits mel; baselines emit waveforms. The vocoder therefore
upper-bounds every fidelity number we can report: if ground-truth mel
through the vocoder already scores below Mimi's reconstruction, the RQ1
table design is lost before training starts (RISK_ASSESSMENT R5).

    python scripts/vocoder_ceiling.py --eval-dir <wavs> [--limit 50] \
        [--device cuda] [--out ceiling.json]

Measures the pretrained Vocos 24 kHz mel vocoder (charactr/vocos-mel-24khz)
as analysis-synthesis: wav -> ITS OWN mel features -> wav -> PESQ/STOI/
SI-SDR vs the original.

NOTE ON MEL CONFIGS: pretrained Vocos consumes ITS feature layout (hop
256), not our 50 Hz hop-480 mel - the number it yields is the ceiling of
"a good mel vocoder at 24 kHz", the right go/no-go signal for R5. The
stage-C vocoder (trained on OUR mel) must then reach comparable ceiling;
re-run this script with --vocoder-ckpt once stage C exists.

Install (login node): pip install vocos  (checkpoint pulls via HF; cache
under $FACT_CACHE/hf as usual).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fact.eval.metrics_audio import pesq_wb, resample, si_sdr, stoi


def load_vocos(device: str):
    try:
        from vocos import Vocos
    except ImportError as e:
        raise SystemExit("pip install vocos (login node; HF cache on $FACT_CACHE)") from e
    return Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import soundfile as sf
    vocos = load_vocos(args.device)
    sr_target = 24_000

    paths = sorted(p for p in args.eval_dir.rglob("*")
                   if p.suffix.lower() in (".wav", ".flac"))[: args.limit]
    if not paths:
        raise SystemExit(f"No audio under {args.eval_dir}")

    scores = {"pesq_wb": [], "stoi": [], "si_sdr": []}
    with torch.no_grad():
        for p in paths:
            wav, sr = sf.read(p, dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            ref = resample(torch.from_numpy(wav), sr, sr_target).to(args.device)
            mel = vocos.feature_extractor(ref.unsqueeze(0))
            hyp = vocos.decode(mel).squeeze(0).cpu()
            ref = ref.cpu()
            scores["pesq_wb"].append(pesq_wb(ref, hyp, sr_target))
            scores["stoi"].append(stoi(ref, hyp, sr_target))
            scores["si_sdr"].append(si_sdr(ref, hyp))

    report = {
        "n_utterances": len(paths),
        "vocoder": "charactr/vocos-mel-24khz (analysis-synthesis ceiling)",
        **{k: {"mean": float(torch.tensor(v).mean()),
               "std": float(torch.tensor(v).std())} for k, v in scores.items()},
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
    mean_pesq = report["pesq_wb"]["mean"]
    # Go/no-go per RISK R5: the ceiling must clear low-bitrate codec range.
    verdict = "GO" if mean_pesq >= 3.0 else "NO-GO: ceiling below low-bitrate codec range"
    print(f"\nR5 verdict: {verdict} (ceiling PESQ-WB {mean_pesq:.2f}; "
          f"Mimi@8books typically ~2-2.5, DAC-24k ~3.5-4.2)")


if __name__ == "__main__":
    main()
