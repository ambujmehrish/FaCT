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
# Site modules are optional: the venv carries its own Python and PyTorch
# wheels bundle CUDA. Set e.g. FACT_MODULES="cuda/12.1" if a job needs one.
if [ -n "${FACT_MODULES:-}" ]; then module purge; module load $FACT_MODULES; fi
source "${FACT_ENV:-$WORK/fact/fact_env.sh}"
export PYTHONUNBUFFERED=1          # progress lines reach the log immediately
export PYTHONUTF8=1                # batch nodes may run an ASCII locale
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

CKPT=${CKPT:?set CKPT=/path/to/step_XXXXXXX.pt}
CONFIG=${CONFIG:-configs/fact_pilot.yaml}
# No default: pilot shards are training data. Full-scale runs: shards/dev.
SHARDS=${SHARDS:?set SHARDS=<held-out shard dir, e.g. ${FACT_DATA:-$FACT_BASE}/shards/dev>}

python scripts/tripwires.py \
    --config "$CONFIG" --ckpt "$CKPT" --shards "$SHARDS" \
    --out "${OUT:-results/tripwires/$(basename "$CKPT" .pt).json}" \
    --device cuda
