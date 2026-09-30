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
module purge
module load python/3.11 cuda
source "$WORK/fact-venv/bin/activate"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python scripts/sweep_lambda_ctc.py \
    --config "${CONFIG:-configs/fact_base.yaml}" \
    --shards "${SHARDS:-$WORK/shards/pilot}" \
    --lambdas "${LAMBDAS:-0.3,1.0,3.0}" \
    --steps "${STEPS:-50000}" \
    --batch-size "${BATCH:-8}" \
    --device cuda \
    --out "results/lambda_ctc_sweep"
