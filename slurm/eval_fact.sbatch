#!/bin/bash
# FaCT reconstruction in the mel domain on the baseline eval set (same mel
# space as eval_baselines.py). Short 1-GPU job per checkpoint.
# Usage: CKPT=<step.pt> OUT=results/fact_recon/<run>.json sbatch slurm/eval_fact.sbatch
#SBATCH --job-name=fact-eval-fact
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs results/fact_recon
# Site modules are optional: the venv carries its own Python and PyTorch
# wheels bundle CUDA. Set e.g. FACT_MODULES="cuda/12.1" if a job needs one.
if [ -n "${FACT_MODULES:-}" ]; then module purge; module load $FACT_MODULES; fi
source "${FACT_ENV:-$WORK/fact/fact_env.sh}"
export PYTHONUNBUFFERED=1          # progress lines reach the log immediately
export PYTHONUTF8=1                # batch nodes may run an ASCII locale
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

CKPT=${CKPT:?set CKPT=/path/to/step_XXXXXXX.pt}
python scripts/eval_fact_recon.py \
    --ckpt "$CKPT" \
    $( [ -n "${EVAL_MANIFEST:-}" ] && echo "--eval-manifest $EVAL_MANIFEST" \
       || echo "--eval-dir ${EVAL_DIR:-$FACT_BASE/data/eval/librispeech_test_clean}" ) \
    --out "${OUT:-results/fact_recon/$(basename "$(dirname "$CKPT")")_$(basename "$CKPT" .pt).json}" \
    --sampler "${SAMPLER:-fm}" --nfe "${NFE:-32}" \
    --device cuda
