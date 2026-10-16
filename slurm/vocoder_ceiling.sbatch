#!/bin/bash
# R5 vocoder ceiling - pretrained Vocos 24 kHz analysis-synthesis on the
# staged eval set. Short 1-GPU job; run before comparing against baselines.
# Prereq (LOGIN NODE, once): scripts/cineca_setup.sh stages the eval set and
# caches charactr/vocos-mel-24khz under $FACT_CACHE.
#SBATCH --job-name=fact-vocoder
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs results
# Site modules are optional: the venv carries its own Python and PyTorch
# wheels bundle CUDA. Set e.g. FACT_MODULES="cuda/12.1" if a job needs one.
if [ -n "${FACT_MODULES:-}" ]; then module purge; module load $FACT_MODULES; fi
source "${FACT_ENV:-$WORK/fact/fact_env.sh}"
export PYTHONUNBUFFERED=1          # progress lines reach the log immediately
export PYTHONUTF8=1                # batch nodes may run an ASCII locale
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python scripts/vocoder_ceiling.py \
    --eval-dir "${EVAL_DIR:-$FACT_BASE/data/eval/librispeech_test_clean}" \
    --limit "${LIMIT:-200}" \
    --device cuda \
    --out "${OUT:-results/vocoder_ceiling.json}"
