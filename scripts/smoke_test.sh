#!/usr/bin/env bash
# End-to-end smoke test on REAL data (LibriSpeech test-clean utterances).
#
# Exercises the exact experiment pipeline, no synthetic data anywhere:
#   download -> manifest -> preprocess (pyworld F0, shard fingerprint)
#   -> stats-derived config (the real mel-normalization workflow)
#   -> stage A training -> stage B shortcut distillation
#   -> tokenize / detokenize / reconstruct / CTC feasibility / probes
#
# Usage:
#   scripts/smoke_test.sh [workdir]              # full run (needs internet
#                                                #   for the download phase)
#   DOWNLOAD_ONLY=1 scripts/smoke_test.sh [dir]  # CINECA login node: fetch
#                                                #   data, then stop
#   scripts/smoke_test.sh [dir]                  # re-run offline: download
#                                                #   phase auto-skips if the
#                                                #   data is already there
# Env knobs: N_UTTS (default 16), STEPS_A (20), STEPS_B (10), WORKERS (4),
#            DEVICE (auto), BATCH (2).
#
# Wall clock: a few minutes on one GPU; CPU-tolerable at the defaults.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORKDIR=${1:-"$ROOT/smoke_work"}
N_UTTS=${N_UTTS:-16}
STEPS_A=${STEPS_A:-20}
STEPS_B=${STEPS_B:-10}
WORKERS=${WORKERS:-4}
BATCH=${BATCH:-2}
DEVICE=${DEVICE:-}
DEVICE_ARG=()
[ -n "$DEVICE" ] && DEVICE_ARG=(--device "$DEVICE")

DATA="$WORKDIR/data"
SHARDS="$WORKDIR/shards"
RUNS="$WORKDIR/runs"
CFG="$WORKDIR/config.yaml"
cd "$ROOT"
mkdir -p "$WORKDIR"

step() { printf '\n=== [%s] %s ===\n' "$1" "$2"; }

step 1/7 "real eval data (LibriSpeech test-clean, $N_UTTS utterances)"
if compgen -G "$DATA/*.wav" > /dev/null; then
    echo "data already present in $DATA - skipping download (offline-safe)"
else
    python scripts/prepare_eval_set.py --out "$DATA" -n "$N_UTTS"
fi
if [ "${DOWNLOAD_ONLY:-0}" = "1" ]; then
    echo "DOWNLOAD_ONLY=1: data staged in $DATA; re-run without it on the compute node."
    exit 0
fi

step 2/7 "manifest (real wav + transcript pairs)"
python -m fact.data.manifest generic --root "$DATA" \
    --out "$WORKDIR/manifest.tsv" --require-text

step 3/7 "preprocess: mel + pyworld F0 -> fingerprinted shards + stats.json"
python -m fact.data.preprocess \
    --manifest "$WORKDIR/manifest.tsv" --out "$SHARDS" \
    --config configs/fact_smoke.yaml \
    --workers "$WORKERS" --min-sec 1.0 --max-sec 30.0

step 4/7 "derive run config from corpus stats (the real workflow)"
python - "$SHARDS" "$CFG" "$STEPS_A" <<'PY'
import json, sys, yaml
shards, out_cfg, steps_a = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = yaml.safe_load(open("configs/fact_smoke.yaml"))
stats = json.load(open(f"{shards}/stats.json"))
assert stats["n_utterances"] > 0, "no utterances survived preprocessing"
cfg["audio"]["mel_mean"] = stats["mel_mean"]
cfg["audio"]["mel_std"] = stats["mel_std"]
cfg["train"]["ckpt_every"] = steps_a  # save exactly at the end of stage A
yaml.safe_dump(cfg, open(out_cfg, "w"), sort_keys=False)
print(f"config -> {out_cfg} (mel_mean={stats['mel_mean']}, "
      f"mel_std={stats['mel_std']}, {stats['hours']*3600:.0f}s of audio)")
PY

step 5/7 "stage A training on real shards ($STEPS_A steps)"
python -m fact.train.stage_a \
    --config "$CFG" --shards "$SHARDS" \
    --ckpt-dir "$RUNS/a" --steps "$STEPS_A" --batch-size "$BATCH" \
    "${DEVICE_ARG[@]}"
CKPT_A=$(ls -t "$RUNS/a"/step_*.pt | head -1)
echo "stage A checkpoint: $CKPT_A"

step 6/7 "stage B shortcut distillation ($STEPS_B steps, tokenizer frozen)"
python -m fact.train.stage_b_shortcut \
    --config "$CFG" --init "$CKPT_A" --shards "$SHARDS" \
    --ckpt-dir "$RUNS/b" --steps "$STEPS_B" --batch-size "$BATCH" \
    "${DEVICE_ARG[@]}"

step 7/7 "roundtrip verification on real audio (tokenize/decode/CTC/probes)"
python scripts/smoke_roundtrip.py \
    --config "$CFG" --ckpt "$CKPT_A" --shards "$SHARDS" "${DEVICE_ARG[@]}"

printf '\nSMOKE TEST PASSED - real-data pipeline is sound.\n'
printf 'Workdir: %s (delete it freely; it is regenerated each run)\n' "$WORKDIR"
