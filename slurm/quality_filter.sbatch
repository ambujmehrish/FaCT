#!/bin/bash
# DNSMOS scan of a manifest's Emilia sidecars (fact.data.quality_filter), one
# array task per part:  MANIFEST=<train.tsv> OUT=<dir> NUM_PARTS=16
#   sbatch --array=0-15 slurm/quality_filter.sbatch
#SBATCH --job-name=fact-qfilter
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                    # smallest allocation on the GPU partition (I/O-bound job)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
mkdir -p logs
source "${FACT_ENV:-$WORK/fact/fact_env.sh}"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1
python -m fact.data.quality_filter scan --manifest "${MANIFEST:?set MANIFEST}" --out "${OUT:?set OUT}" \
    --num-parts "${NUM_PARTS:?set NUM_PARTS}" --part "${SLURM_ARRAY_TASK_ID:-0}" ${EXTRA:-}
