#!/bin/bash
# Sampling sweep scored on WAVEFORM metrics (vocoder + Whisper WER) on the
# FULL Emilia dev set - the model-selection set, never a test set. The
# earlier mel-L1 sweep (submit_dev_sweep.sh) stopped at the edge of its grid
# (T=0.5 was its lowest and best temperature) and never measured WER.
#
#   bash scripts/submit_wer_sweep.sh                         # main_c8_rd05 @ 400K
#   RUNS="main_c8_rd05 main_c8_rd03 main_c16_rd05" bash scripts/submit_wer_sweep.sh
#   RUNS=ft_c8_rd05_distill1 STEP=0100000 TEMPS=0.5 CFGS=1.5 bash scripts/submit_wer_sweep.sh
#
# Output: results/baseline_recon_dev/fact_<run>@<step>_<view>_T<t>_G<g>.json
# Selection rule (fixed BEFORE any result, scripts/select_wer_setting.py):
#   among this run's full-view settings, the lowest dev corpus WER whose dev
#   mel-L1 is at most 0.02 above the current setting's (T=0.5, NFE=32,
#   CFG=1.5); the native tokens-only view is reported at that setting.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
D=${D:-/leonardo_scratch/large/userexternal/amehrish/fact}
ACCOUNT=${ACCOUNT:-AIFAC_S07_041}
RUNS=${RUNS:-main_c8_rd05}
STEP=${STEP:-0400000}
DEV=${DEV:-$D/manifests/dev.tsv}
TEMPS=${TEMPS:-"0.5 0.3 0.0"}
CFGS=${CFGS:-"1.5 2.0 3.0"}
NFE=${NFE:-32}
VOC=${VOC:-$(ls "$D"/runs/vocoder/step_*.pt 2>/dev/null | sort | tail -1)}
[ -f "$VOC" ] || { echo "no vocoder checkpoint (set VOC=)"; exit 1; }
[ -f "$DEV" ] || { echo "missing $DEV"; exit 1; }
mkdir -p logs results/baseline_recon_dev
queued=$(squeue --me -h -o "%j" 2>/dev/null || true)
echo "vocoder: $VOC"

for R in $RUNS; do
    CK=$D/runs/$R/step_$STEP.pt
    [ -f "$CK" ] || { echo "$R: no step_$STEP.pt, skipped"; continue; }
    native=$(python - "$CK" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=True)["config"]["bottleneck"]
print("discrete_only" if c.get("residual_dropout", 0) > 0 else "tokens_prior")
PY
)
    echo "== $R @ $STEP (views full,$native)"
    for T in $TEMPS; do for G in $CFGS; do for V in full $native; do
        label="fact_${R}@${STEP}_${V}_T${T}_G${G}"
        out=results/baseline_recon_dev/$label.json
        name=fact-eb-dev-$label
        if [ -e "$out" ]; then echo "  done    $label"; continue; fi
        if grep -qx "$name" <<<"$queued"; then echo "  queued  $label"; continue; fi
        id=$(BASELINES="fact:ckpt=$CK:vocoder=$VOC:view=$V:nfe=$NFE:temperature=$T:cfg_scale=$G:spk_ref=other:label=$label" \
             EVAL_MANIFEST="$DEV" OUT=results/baseline_recon_dev \
             EXTRA="--wer-model large-v3 --utmos --partner-subset" \
             sbatch --parsable --account="$ACCOUNT" --job-name="$name" --time=12:00:00 slurm/eval_baselines.sbatch)
        echo "  SUBMIT  $label -> $id"
    done; done; done
done
echo "When done: python scripts/select_wer_setting.py --run <run> [--step $STEP]"
