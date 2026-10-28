#!/bin/bash
# Check that every TTS token shard loads and the codecs agree on item counts
# (fact.tts.check_shards); unreadable shards are moved to $QUARANTINE so a
# tokenize rerun (NUM_PARTS=1, it skips existing shards) writes them again.
#   DIRS="<tok>/fact_b1 <tok>/mimi2_b1" OUT=results/tts/shard_check.json \
#   QUARANTINE=<tok>/quarantine sbatch slurm/tts_check_shards.sbatch
#SBATCH --job-name=fact-tts-check
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                    # smallest allocation on the GPU partition (I/O-bound job)
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
if [ -n "${FACT_MODULES:-}" ]; then module purge; module load $FACT_MODULES; fi
source "${FACT_ENV:-$WORK/fact/fact_env.sh}"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 OMP_NUM_THREADS=1
python -m fact.tts.check_shards --dirs ${DIRS:?set DIRS} --out "${OUT:?set OUT}" \
    ${QUARANTINE:+--quarantine "$QUARANTINE"} --threads "$SLURM_CPUS_PER_TASK"
