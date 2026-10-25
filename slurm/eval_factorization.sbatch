#!/bin/bash
# Factorization evaluation (scripts/eval_factorization.py), one shard per job.
#   DECODER=<runs>/wavedec_main_c8_rd05/step_0250000.pt OUT=results/factorization/x_shard0.json \
#   SHARD=0 NUM_SHARDS=8 EVAL_DIR=<eval>/librispeech_test_clean_full \
#   EXTRA="--wer-model large-v3 --sim" sbatch --account=<acct> slurm/eval_factorization.sbatch
#SBATCH --job-name=fact-factorization
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
if [ -n "${FACT_MODULES:-}" ]; then module purge; module load $FACT_MODULES; fi
source "${FACT_ENV:-$WORK/fact/fact_env.sh}"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python scripts/eval_factorization.py \
    --decoder "${DECODER:?set DECODER=<stage-D step.pt>}" \
    --eval-dir "${EVAL_DIR:-$FACT_BASE/data/eval/librispeech_test_clean_full}" \
    --out "${OUT:?set OUT}" \
    --shard "${SHARD:-0}" --num-shards "${NUM_SHARDS:-1}" \
    --device cuda ${EXTRA:-} ${BASELINE_SPECS:+--baselines $BASELINE_SPECS}
