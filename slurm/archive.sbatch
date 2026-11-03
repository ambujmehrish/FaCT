#!/bin/bash
# Archive the checkpoints behind the reported results from SCRATCH (purged
# after ~40 days) to WORK, with pip freeze and Hugging Face revisions
# (scripts/archive_artifacts.py). CPU/IO-bound; smallest GPU allocation.
#   OUT=<WORK>/fact/archive/<date> [LIST=ops/archive/checkpoints.txt] sbatch slurm/archive.sbatch
#SBATCH --job-name=fact-archive
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                    # smallest allocation on the GPU partition (I/O-bound job)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
if [ -n "${FACT_MODULES:-}" ]; then module purge; module load $FACT_MODULES; fi
source "${FACT_ENV:-$WORK/fact/fact_env.sh}"
source "${SLURM_SUBMIT_DIR:-.}/slurm/_provenance.sh"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python scripts/archive_artifacts.py --runs-dir "${FACT_RUNS:-$FACT_BASE/runs}" \
    --out "${OUT:?set OUT=<archive dir on WORK>}" --list "${LIST:-ops/archive/checkpoints.txt}"
