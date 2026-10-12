#!/bin/bash
# Gate tripwires on a checkpoint (RISK R1/R2) - run after every pilot
# checkpoint lands. Short 1-GPU job.
# Usage: CKPT=$FACT_RUNS/pilot/step_0025000.pt CONFIG=... SHARDS=... sbatch slurm/tripwires.sbatch
#SBATCH --job-name=fact-tripwires
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs results/tripwires
module purge
module load python/3.11 cuda
source "${FACT_ENV:-$WORK/fact/fact_env.sh}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

CKPT=${CKPT:?set CKPT=/path/to/step_XXXXXXX.pt}
CONFIG=${CONFIG:-configs/fact_pilot.yaml}
SHARDS=${SHARDS:-${FACT_DATA:-$FACT_BASE}/shards/pilot}

python scripts/tripwires.py \
    --config "$CONFIG" --ckpt "$CKPT" --shards "$SHARDS" \
    --out "results/tripwires/$(basename "$CKPT" .pt).json" \
    --device cuda
