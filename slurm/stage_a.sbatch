#!/bin/bash
# Stage A training - 4x A100-64GB (one Leonardo boost node), DDP via torchrun.
# The plan's batch was sized for 8x A100-80GB; on 4x64GB keep the effective
# batch with train.grad_accum (see configs) - halve per-GPU batch, accum x4.
#SBATCH --job-name=fact-stage-a
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
# Leonardo caps wall time; chain runs with --resume:
#   sbatch --dependency=afterok:<jobid> slurm/stage_a.sbatch

set -euo pipefail
mkdir -p logs
module purge
module load python/3.11 cuda
source "$WORK/fact-venv/bin/activate"

export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8
CONFIG=${CONFIG:-configs/fact_base.yaml}
RUN=${RUN:-fact_base}
RESUME_ARG=""
LATEST=$(ls -t "$WORK/runs/$RUN"/step_*.pt 2>/dev/null | head -1 || true)
[ -n "$LATEST" ] && RESUME_ARG="--resume $LATEST"

torchrun --standalone --nproc_per_node=4 -m fact.train.stage_a \
    --config "$CONFIG" \
    --shards "$WORK/shards/train" \
    --ckpt-dir "$WORK/runs/$RUN" \
    --batch-size 8 \
    --num-workers 7 \
    $RESUME_ARG
# Baseline arms: CONFIG=configs/baseline_single_fsq.yaml RUN=single_fsq sbatch ...
#                CONFIG=configs/baseline_vae.yaml        RUN=vae        sbatch ...
