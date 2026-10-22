#!/bin/bash
# Map preprocessed shard items to their audio files (scripts/build_audio_index.py)
# for the SSL-input run (#4). CPU work; one small job per shard directory.
#   SHARDS=<shards>/train_v2 MANIFESTS='<manifests>/train_c[0-9][0-9][0-9].tsv' sbatch slurm/build_audio_index.sbatch
#   SHARDS=<shards>/dev MANIFESTS=<manifests>/dev.tsv EXTRA=--no-prefix sbatch slurm/build_audio_index.sbatch
#SBATCH --job-name=fact-audio-index
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                    # smallest allocation on the GPU partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
source "${FACT_ENV:-$WORK/fact/fact_env.sh}"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1
python scripts/build_audio_index.py --shards "${SHARDS:?set SHARDS}" \
    --manifests "${MANIFESTS:?set MANIFESTS}" ${EXTRA:-}
