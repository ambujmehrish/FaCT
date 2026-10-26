#!/bin/bash
# Zero-shot TTS evaluation (scripts/eval_tts.py), one shard per job.
#   LM=<runs>/tts_fact/step_0200000.pt WAVE_DECODER=<stage-D step.pt> OUT=results/tts/x_shard0.json \
#   EVAL_DIR=<LS test-clean> | EVAL_MANIFEST=<Emilia dev.tsv>  SHARD=0 NUM_SHARDS=8 \
#   EXTRA="--wer-model large-v3 --temperature 1.0 --top-k 50" sbatch slurm/eval_tts.sbatch
#SBATCH --job-name=fact-eval-tts
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
if [ -n "${FACT_MODULES:-}" ]; then module purge; module load $FACT_MODULES; fi
source "${FACT_ENV:-$WORK/fact/fact_env.sh}"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
if [ -n "${EVAL_MANIFEST:-}" ]; then SRC="--eval-manifest $EVAL_MANIFEST"; else SRC="--eval-dir ${EVAL_DIR:?set EVAL_DIR or EVAL_MANIFEST}"; fi
python scripts/eval_tts.py --lm "${LM:?set LM}" ${WAVE_DECODER:+--wave-decoder "$WAVE_DECODER"} \
    $SRC --out "${OUT:?set OUT}" --shard "${SHARD:-0}" --num-shards "${NUM_SHARDS:-1}" \
    --device cuda ${EXTRA:-}
