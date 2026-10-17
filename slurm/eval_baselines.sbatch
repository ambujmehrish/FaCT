#!/bin/bash
# Public-checkpoint baseline reconstruction eval - CINECA Leonardo.
# Prereq (LOGIN NODE, once): scripts/cineca_setup.sh - it stages all
# checkpoints + the eval set offline-ready under $FACT_BASE (never $HOME)
# and writes the fact_env.sh this job sources.
#SBATCH --job-name=fact-eval-base
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CHANGE_ME            # your CINECA project account
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
# Site modules are optional: the venv carries its own Python and PyTorch
# wheels bundle CUDA. Set e.g. FACT_MODULES="cuda/12.1" if a job needs one.
if [ -n "${FACT_MODULES:-}" ]; then module purge; module load $FACT_MODULES; fi
source "${FACT_ENV:-$WORK/fact/fact_env.sh}"
export PYTHONUNBUFFERED=1          # progress lines reach the log immediately
export PYTHONUTF8=1                # batch nodes may run an ASCII locale

export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python scripts/eval_baselines.py \
    --baselines "${BASELINES:-mimi,dac}" \
    --eval-dir "${EVAL_DIR:-$FACT_BASE/data/eval/librispeech_test_clean}" \
    --out "${OUT:-results/baseline_recon}" \
    --device cuda \
    --threads "$SLURM_CPUS_PER_TASK"
# Add: --wer-model large-v3   (after caching whisper on the login node)
# Bitrate-matched variants, e.g. BASELINES="mimi:n_streams=2,mimi:n_streams=4,dac:n_quantizers=1,dac:n_quantizers=2"
# Full test-clean (2,620 utts, ~4 h for 6 specs): EVAL_DIR=$FACT_BASE/data/eval/librispeech_test_clean_full
#   OUT=results/baseline_recon_full sbatch --time=08:00:00 ...
