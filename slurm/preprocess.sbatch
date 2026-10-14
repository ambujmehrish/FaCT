#!/bin/bash
# Shard preprocessing (mel + F0). CPU-bound but submitted as a normal job so
# login nodes are never loaded; uses the full core count of one node.
# Manifests must already exist (built on the login node where data lives).
#SBATCH --job-name=fact-preprocess
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                    # smallest allocation on the GPU partition
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
# Site modules are optional: the venv carries its own Python and PyTorch
# wheels bundle CUDA. Set e.g. FACT_MODULES="cuda/12.1" if a job needs one.
if [ -n "${FACT_MODULES:-}" ]; then module purge; module load $FACT_MODULES; fi
source "${FACT_ENV:-$WORK/fact/fact_env.sh}"
export PYTHONUNBUFFERED=1          # progress lines reach the log immediately
export OMP_NUM_THREADS=1                # workers are process-parallel; keep BLAS at 1

python -m fact.data.preprocess \
    --manifest "${MANIFEST:-${FACT_DATA:-$FACT_BASE}/manifests/train.tsv}" \
    --out "${OUT:-${FACT_DATA:-$FACT_BASE}/shards/train}" \
    --config "${CONFIG:-configs/fact_base.yaml}" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --min-sec 1.0 --max-sec 30.0 \
    ${EXTRA:-}
# EXTRA passes chunked-staging / storage flags, e.g.
#   EXTRA="--shard-prefix c000_ --skip-stats --mel-dtype int8"
# Pilot slice: MANIFEST=${FACT_DATA:-$FACT_BASE}/manifests/pilot.tsv OUT=${FACT_DATA:-$FACT_BASE}/shards/pilot \
#              CONFIG=configs/fact_pilot.yaml sbatch slurm/preprocess.sbatch
# Resumable: resubmit the same command after a wall-time kill.
