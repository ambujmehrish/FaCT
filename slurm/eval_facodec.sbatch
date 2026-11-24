#!/bin/bash
# FACodec head-to-head (scripts/eval_facodec.py), one shard per job.
#   OUT=results/facodec/ls_shard0.json \
#   SHARD=0 NUM_SHARDS=8 EVAL_DIR=<eval>/librispeech_test_clean_full   (or EVAL_MANIFEST=<tsv>) \
#   EXTRA="--wer-model large-v3 --utmos" sbatch --account=<acct> slurm/eval_vc.sbatch
#SBATCH --job-name=fact-facodec
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
if [ -n "${FACT_MODULES:-}" ]; then module purge; module load $FACT_MODULES; fi
source "${FACT_ENV:-$WORK/fact/fact_env.sh}"
source "${SLURM_SUBMIT_DIR:-.}/slurm/_provenance.sh"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

if [ -n "${EVAL_MANIFEST:-}" ]; then SRC=(--eval-manifest "$EVAL_MANIFEST");
else SRC=(--eval-dir "${EVAL_DIR:-$FACT_BASE/data/eval/librispeech_test_clean_full}"); fi
python scripts/eval_facodec.py "${SRC[@]}" \
    --out "${OUT:?set OUT}" \
    --shard "${SHARD:-0}" --num-shards "${NUM_SHARDS:-1}" \
    --device cuda ${EXTRA:-}
