#!/bin/bash
# Lambda_ctc sweep at pilot scale (RISK R3 / ablation #2) - run BEFORE the
# main stage-A arms. Sequential arms on one GPU; the sweep script fixes
# seeds and batch order so arms differ only in ctc.weight.
# Prereq: pilot shards preprocessed (e.g. 5K-h Emilia slice).
#SBATCH --job-name=fact-lctc-sweep
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
# Site modules are optional: the venv carries its own Python and PyTorch
# wheels bundle CUDA. Set e.g. FACT_MODULES="cuda/12.1" if a job needs one.
if [ -n "${FACT_MODULES:-}" ]; then module purge; module load $FACT_MODULES; fi
source "${FACT_ENV:-$WORK/fact/fact_env.sh}"
export PYTHONUNBUFFERED=1          # progress lines reach the log immediately
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python scripts/sweep_lambda_ctc.py \
    --config "${CONFIG:-configs/fact_base.yaml}" \
    --shards "${SHARDS:-${FACT_DATA:-$FACT_BASE}/shards/pilot}" \
    --lambdas "${LAMBDAS:-0.3,1.0,3.0}" \
    --steps "${STEPS:-50000}" \
    --batch-size "${BATCH:-8}" \
    --num-workers "${NUM_WORKERS:-6}" \
    --device cuda \
    --out "${OUT:-results/lambda_ctc_sweep}" \
    ${EXTRA:-}
# Re-evaluate saved checkpoints only: EXTRA="--eval-only --eval-shards <dev shards>"
# One job per lambda (each fits the 24 h limit on its own), e.g.:
#   for L in 0.3 1.0 3.0; do LAMBDAS=$L OUT=results/lambda_ctc_sweep/lam_$L \
#     CONFIG=configs/fact_pilot.yaml sbatch -A <account> slurm/sweep_lambda_ctc.sbatch; done
