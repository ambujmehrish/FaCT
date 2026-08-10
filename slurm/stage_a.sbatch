#!/bin/bash
# Stage A training - multi-node DDP on CINECA Leonardo.
# Default: 2 nodes x 4 A100-64GB = 8 GPUs (the plan's full compute budget).
# Set --nodes=1 for a single-node run; the launcher below handles both.
#SBATCH --job-name=fact-stage-a
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1            # one torchrun per node; it spawns 4 workers
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
# Leonardo caps wall time; chain runs (auto-resumes from newest checkpoint):
#   sbatch --dependency=afterok:<jobid> slurm/stage_a.sbatch

set -euo pipefail
mkdir -p logs
module purge
module load python/3.11 cuda            # pin exact versions from `module avail`
source "$WORK/fact-venv/bin/activate"

export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8                # 32 cores / 4 GPU workers per node
# NCCL over Leonardo's InfiniBand usually autodetects; uncomment to debug:
# export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))   # unique per job, avoids collisions

CONFIG=${CONFIG:-configs/fact_base.yaml}
RUN=${RUN:-fact_base}
RESUME_ARG=""
LATEST=$(ls -t "$WORK/runs/$RUN"/step_*.pt 2>/dev/null | head -1 || true)
[ -n "$LATEST" ] && RESUME_ARG="--resume $LATEST"

# srun starts one torchrun per node; c10d rendezvous joins them into one
# 8-process world (RANK/WORLD_SIZE/LOCAL_RANK are set per worker).
srun torchrun \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc_per_node=4 \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    -m fact.train.stage_a \
    --config "$CONFIG" \
    --shards "$WORK/shards/train" \
    --ckpt-dir "$WORK/runs/$RUN" \
    --batch-size 8 \
    --num-workers 7 \
    $RESUME_ARG
# 8 x A100-64GB vs the plan's 8 x 80GB: if batch-size 8/GPU OOMs, drop to 6
# and set train.grad_accum: 2 in the config to keep the effective batch.
# Baseline arms: CONFIG=configs/baseline_single_fsq.yaml RUN=single_fsq sbatch ...
#                CONFIG=configs/baseline_vae.yaml        RUN=vae        sbatch ...
# Ablations:     CONFIG=configs/ablation_rvq.yaml        RUN=rvq        sbatch ...
#                CONFIG=configs/ablation_noncausal.yaml  RUN=noncausal  sbatch ...
