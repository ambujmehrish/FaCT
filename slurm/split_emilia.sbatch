#!/bin/bash
# Group-disjoint train/dev/test split of the full Emilia manifest, plus the
# train chunks for parallel preprocessing (scripts/split_emilia.py). One
# pass over the 4.6 GB manifest; CPU-only, fits the serial partition.
#SBATCH --job-name=fact-split
#SBATCH --partition=lrd_all_serial
#SBATCH --account=CHANGE_ME
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "${FACT_ENV:-$WORK/fact/fact_env.sh}"
export PYTHONUNBUFFERED=1          # progress lines reach the log immediately
export PYTHONUTF8=1                # batch nodes may run an ASCII locale

M=${MANIFESTS:-${FACT_DATA:-$FACT_BASE}/manifests}
python scripts/split_emilia.py \
    --manifest "${MANIFEST:-$M/emilia_en.tsv}" \
    --out-dir "${OUT_DIR:-$M}" \
    --chunks "${CHUNKS:-36}" \
    --heldout-per-10k "${HELDOUT_PER_10K:-100}" \
    ${EXTRA:-}
