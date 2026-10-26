#!/bin/bash
# TTS LM training (fact.tts.train) - 2 nodes x 4 GPUs, auto-resumes from the
# newest checkpoint in $FACT_RUNS/$RUN; chain parts with afterany.
#   TOKENS=<dir> REQUIRE="<other dirs>" RUN=tts_fact sbatch slurm/tts_train.sbatch
#SBATCH --job-name=fact-tts-train
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
if [ -n "${FACT_MODULES:-}" ]; then module purge; module load $FACT_MODULES; fi
source "${FACT_ENV:-$WORK/fact/fact_env.sh}"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
RUN=${RUN:?set RUN}
CKPT_DIR=${FACT_RUNS:-${FACT_DATA:-$FACT_BASE}/runs}/$RUN
if [ -n "${REQUIRE_RESUME:-}" ] && ! ls "$CKPT_DIR"/step_*.pt >/dev/null 2>&1; then
    echo "REQUIRE_RESUME set but no checkpoint in $CKPT_DIR"; exit 1
fi
echo "RUN=$RUN tokens=${TOKENS:-} require=${REQUIRE:-} EXTRA=${EXTRA:-}"
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
srun torchrun --nnodes="$SLURM_JOB_NUM_NODES" --nproc_per_node=4 \
    --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    -m fact.tts.train --tokens "${TOKENS:?set TOKENS}" ${REQUIRE:+--require $REQUIRE} \
    --ckpt-dir "$CKPT_DIR" --workers 6 ${EXTRA:-}
