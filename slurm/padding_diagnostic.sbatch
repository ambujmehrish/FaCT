#!/bin/bash
# Padding diagnostic (audit 2026-11-03, C): training losses and tokens with
# padding visible (pre-fix) vs hidden (= evaluation), on dev shards.
# Usage: CKPTS="label=<step.pt>,label=<step.pt>" SHARDS=<dev shards> sbatch slurm/padding_diagnostic.sbatch
#SBATCH --job-name=fact-paddiag
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
if [ -n "${FACT_MODULES:-}" ]; then module purge; module load $FACT_MODULES; fi
source "${FACT_ENV:-$WORK/fact/fact_env.sh}"
source "${SLURM_SUBMIT_DIR:-.}/slurm/_provenance.sh"
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

CKPTS=${CKPTS:?set CKPTS=label=<step.pt>[,label=<step.pt>...]}
SHARDS=${SHARDS:?set SHARDS=<held-out shard dir, e.g. ${FACT_DATA:-$FACT_BASE}/shards/dev>}
OUT=${OUT:-results/padding_diagnostic}
# Fail early: every checkpoint must exist before any model is loaded.
IFS=',' read -ra PAIRS <<< "$CKPTS"
for p in "${PAIRS[@]}"; do
    [ -f "${p#*=}" ] || { echo "missing checkpoint ${p#*=}"; exit 1; }
done
for p in "${PAIRS[@]}"; do
    python scripts/padding_diagnostic.py --label "${p%%=*}" --ckpt "${p#*=}" \
        --shards "$SHARDS" --out "$OUT" --n-batches "${N_BATCHES:-200}" --num-workers 6 --device cuda
done
