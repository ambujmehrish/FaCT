#!/bin/bash
# Semantic distillation into the causal content tokens (distill.*, Mimi's
# recipe: frozen WavLM-large layer 6 as a TRAINING-ONLY teacher; the
# tokenizer stays mel-input and causal at inference). Every arm uses exactly
# the base run's config (read from its checkpoint) plus the listed --set.
#
#   bash scripts/submit_distill_runs.sh             # all arms below
#   ARMS="ft_w1 ft_ctrl" bash scripts/submit_distill_runs.sh
#
# Arms (BASE = main_c8_rd05, the dev-selected 0.42 kbps model):
#   ft_w1   fine-tune BASE@400K for 100K steps, distill.weight=1
#   ft_w3   same, distill.weight=3
#   ft_ctrl same 100K fine-tune with distillation OFF - the control that
#           separates the distillation effect from 100K extra steps
#   scr_w1  BASE's config + distill.weight=1 from scratch, 400K steps
# Fine-tunes: peak LR 1e-4 (BASE ended its cosine at ~0), 2K warmup.
# Each arm is a chain of 24 h jobs (2 nodes x 4 GPUs) that auto-resume.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
D=${D:-/leonardo_scratch/large/userexternal/amehrish/fact}
ACCOUNT=${ACCOUNT:-AIFAC_S07_041}
BASE=${BASE:-main_c8_rd05}
CK=$D/runs/$BASE/step_0400000.pt
SHARDS=${SHARDS:-$D/shards/train_v2}
ARMS=${ARMS:-"ft_w1 ft_w3 ft_ctrl scr_w1"}
[ -f "$CK" ] || { echo "missing $CK"; exit 1; }
[ -d "$SHARDS/audio_index" ] || { echo "missing $SHARDS/audio_index (scripts/build_audio_index.py)"; exit 1; }
FT="--init-from $CK --steps 100000 --set train.lr=0.0001 --set train.warmup_steps=2000"
queued=$(squeue --me -h -o "%j" 2>/dev/null || true)

chain() {  # run-name parts extra
    local run=$1 parts=$2 extra=$3 name="fact-${1}" j
    if grep -qx "$name" <<<"$queued"; then echo "  queued  $run"; return; fi
    j=$(CONFIG="$CK" RUN="$run" SHARDS="$SHARDS" EXTRA="$extra" \
        sbatch --parsable --account="$ACCOUNT" --job-name="$name" slurm/stage_a.sbatch)
    echo "  SUBMIT  $run part1 -> $j"
    for p in $(seq 2 "$parts"); do
        j=$(CONFIG="$CK" RUN="$run" SHARDS="$SHARDS" EXTRA="$extra" REQUIRE_RESUME=1 \
            sbatch --parsable --account="$ACCOUNT" --job-name="$name" \
            --dependency=afterany:"$j" slurm/stage_a.sbatch)
        echo "  SUBMIT  $run part$p -> $j"
    done
}

for A in $ARMS; do
    case $A in
        ft_w1)   chain "ft_${BASE#main_}_distill1" 2 "$FT --set distill.weight=1.0" ;;
        ft_w3)   chain "ft_${BASE#main_}_distill3" 2 "$FT --set distill.weight=3.0" ;;
        ft_ctrl) chain "ft_${BASE#main_}_ctrl"     2 "$FT" ;;
        scr_w1)  chain "${BASE#main_}_distill1"    6 "--set distill.weight=1.0" ;;
        *) echo "unknown arm $A"; exit 1 ;;
    esac
done
