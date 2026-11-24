#!/bin/bash
# Real-time factor of encode+decode for FaCT and the baselines (scripts/measure_rtf.py).
#   SYSTEMS="<spec> <spec> ..." OUT=results/rtf/rtf.json sbatch --account=<acct> slurm/measure_rtf.sbatch
#SBATCH --job-name=fact-rtf
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
if [ -n "${FACT_MODULES:-}" ]; then module purge; module load $FACT_MODULES; fi
source "${FACT_ENV:-$WORK/fact/fact_env.sh}"
source "${SLURM_SUBMIT_DIR:-.}/slurm/_provenance.sh"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

read -r -a SYS <<< "${SYSTEMS:?set SYSTEMS}"
python scripts/measure_rtf.py \
    --eval-dir "${EVAL_DIR:-$FACT_BASE/data/eval/librispeech_test_clean_full}" \
    --n "${N:-100}" --devices "${DEVICES:-cuda,cpu}" --cpu-threads "$SLURM_CPUS_PER_TASK" \
    --out "${OUT:?set OUT}" --systems "${SYS[@]}"
