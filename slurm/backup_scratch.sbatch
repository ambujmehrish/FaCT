#!/bin/bash
# Copy what a rebuttal needs from SCRATCH (purged after ~40 days) to WORK:
# full checkpoints (with optimizer state), the eval manifests and their
# audio, dev shards and model caches (scripts/backup_scratch.py). Files
# already outside SCRATCH are only listed. I/O-bound; smallest GPU allocation.
#   OUT=<WORK dir> [LIST=ops/archive/rebuttal_checkpoints.txt] sbatch slurm/backup_scratch.sbatch
#SBATCH --job-name=fact-backup
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                    # smallest allocation on the GPU partition (I/O-bound job)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
if [ -n "${FACT_MODULES:-}" ]; then module purge; module load $FACT_MODULES; fi
source "${FACT_ENV:-$WORK/fact/fact_env.sh}"
source "${SLURM_SUBMIT_DIR:-.}/slurm/_provenance.sh"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DATA=${FACT_DATA:-$FACT_BASE}
args=(--out "${OUT:?set OUT=<backup dir on WORK>}" --scratch-root "${SCRATCH_ROOT:-/leonardo_scratch}"
      --runs-dir "${FACT_RUNS:-$FACT_BASE/runs}" --checkpoints "${LIST:-ops/archive/rebuttal_checkpoints.txt}"
      --manifest "$DATA/manifests/dev.tsv" --manifest "$DATA/manifests/test.tsv"
      --dir "$DATA/shards/dev")
[ -n "${FACT_CACHE:-}" ] && args+=(--dir "$FACT_CACHE")
[ -n "${HF_HOME:-}" ] && args+=(--dir "$HF_HOME")
echo "backup: DATA=$DATA FACT_CACHE=${FACT_CACHE:-unset} HF_HOME=${HF_HOME:-unset}"
du -sh "$DATA/shards/dev" "$DATA/manifests" ${FACT_CACHE:+"$FACT_CACHE"} 2>/dev/null || true
python scripts/backup_scratch.py "${args[@]}"
du -sh "$OUT"
