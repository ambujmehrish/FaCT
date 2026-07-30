#!/bin/bash
# Public-checkpoint baseline reconstruction eval - CINECA Leonardo.
# Prereqs (LOGIN NODE, once - compute nodes have no internet):
#   export HF_HOME=$WORK/hf_cache
#   python scripts/download_checkpoints.py --baselines mimi,dac
#   python scripts/prepare_eval_set.py --out $WORK/data/eval/librispeech_test_clean -n 200
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
module purge
module load python/3.11 cuda            # adjust to `module avail` on Leonardo
source "$WORK/fact-venv/bin/activate"

export HF_HOME="$WORK/hf_cache"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python scripts/eval_baselines.py \
    --baselines mimi,dac \
    --eval-dir "$WORK/data/eval/librispeech_test_clean" \
    --out results/baseline_recon \
    --device cuda \
    --threads "$SLURM_CPUS_PER_TASK"
# Add: --wer-model large-v3   (after caching whisper on the login node)
