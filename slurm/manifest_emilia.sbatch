#!/bin/bash
# Emilia manifest scan as a chain of short jobs.
# lrd_all_serial caps a job at 4 h and a full EN scan takes ~15 h, so queue
# several copies with --dependency; each one resumes from the folders the
# previous one saved, and copies that start after the manifest is built exit
# immediately. See docs/PILOT_RUNBOOK.md step 2.
#SBATCH --job-name=fact-manifest
#SBATCH --partition=lrd_all_serial
#SBATCH --account=CHANGE_ME
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "${FACT_ENV:-$WORK/fact/fact_env.sh}"
export PYTHONUNBUFFERED=1          # progress lines reach the log immediately

ROOT=${ROOT:-${FACT_DATA:-$FACT_BASE}/data/raw/emilia/EN}
OUT=${OUT:-${FACT_DATA:-$FACT_BASE}/manifests/emilia_en.tsv}

if [ -f "$OUT" ]; then
    echo "$OUT already exists ($(wc -l < "$OUT") lines); nothing to do."
    exit 0
fi

python -m fact.data.manifest emilia --workers "${WORKERS:-32}" \
    --root "$ROOT" --out "$OUT" --require-text
