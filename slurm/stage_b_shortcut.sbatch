#!/bin/bash
# Stage B: shortcut-distill the decoder (encoder+bottleneck frozen).
# Decoder-only fine-tune - one node (4 GPUs) is enough; same launcher
# pattern as stage_a.sbatch, so raise --nodes if you want it faster.
#SBATCH --job-name=fact-stage-b
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
module purge
module load python/3.11 cuda
source "$WORK/fact-venv/bin/activate"

export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

CONFIG=${CONFIG:-configs/fact_base.yaml}
RUN=${RUN:-fact_base}
INIT=${INIT:?set INIT=/path/to/stage_a_checkpoint.pt}

srun torchrun \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc_per_node=4 \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    -m fact.train.stage_b_shortcut \
    --config "$CONFIG" \
    --init "$INIT" \
    --shards "$WORK/shards/train" \
    --ckpt-dir "$WORK/runs/${RUN}_stage_b" \
    --steps 20000 \
    --batch-size 8 \
    --num-workers 7
