#!/bin/bash
# Tokenize the training shards for the TTS LMs (fact.tts.tokenize), one array
# task per part:  CODEC=fact|mimi OUT=<dir> NUM_PARTS=64 [FACT_CKPT=...]
#   sbatch --array=0-63 slurm/tts_tokenize.sbatch      (part = SLURM_ARRAY_TASK_ID)
#SBATCH --job-name=fact-tts-tok
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
mkdir -p logs
if [ -n "${FACT_MODULES:-}" ]; then module purge; module load $FACT_MODULES; fi
source "${FACT_ENV:-$WORK/fact/fact_env.sh}"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -m fact.tts.tokenize \
    --codec "${CODEC:?set CODEC=fact|mimi}" \
    ${FACT_CKPT:+--fact-ckpt "$FACT_CKPT"} \
    --shards "${SHARDS:-${FACT_DATA:-$FACT_BASE}/shards/train_v2}" \
    --out "${OUT:?set OUT}" \
    --num-parts "${NUM_PARTS:?set NUM_PARTS}" --part "${SLURM_ARRAY_TASK_ID:-0}" \
    --device cuda ${EXTRA:-}
