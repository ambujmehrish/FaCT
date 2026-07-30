#!/bin/bash
# Shard preprocessing (mel + F0). CPU-bound but submitted as a normal job so
# login nodes are never loaded; uses the full core count of one node.
# Manifests must already exist (built on the login node where data lives).
#SBATCH --job-name=fact-preprocess
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                    # smallest allocation on the GPU partition
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
module purge
module load python/3.11
source "$WORK/fact-venv/bin/activate"
export OMP_NUM_THREADS=1                # workers are process-parallel; keep BLAS at 1

python -m fact.data.preprocess \
    --manifest "$WORK/manifests/train.tsv" \
    --out "$WORK/shards/train" \
    --config configs/fact_base.yaml \
    --workers "$SLURM_CPUS_PER_TASK" \
    --min-sec 1.0 --max-sec 30.0
