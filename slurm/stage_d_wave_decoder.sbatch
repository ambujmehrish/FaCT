#!/bin/bash
# Stage D (#2): GAN waveform decoder on frozen FaCT tokens - 2 nodes x 4 GPUs.
#   FACT_CKPT=<runs>/main_c8_rd05/step_0400000.pt RUN=wavedec_main_c8_rd05 \
#     sbatch --account=<acct> slurm/stage_d_wave_decoder.sbatch
# Auto-resumes from the newest checkpoint in <runs>/$RUN; chain with afterany.
#SBATCH --job-name=fact-wavedec
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
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1

D=${FACT_DATA:-$FACT_BASE}
FACT_CKPT=${FACT_CKPT:?set FACT_CKPT=<frozen FaCT step.pt>}
RUN=${RUN:?set RUN}
CKPT_DIR=${FACT_RUNS:-$D/runs}/$RUN
RESUME_ARG=""
LATEST=$(ls "$CKPT_DIR"/step_*.pt 2>/dev/null | sort | tail -1 || true)
[ -n "$LATEST" ] && RESUME_ARG="--resume $LATEST"
if [ -z "$LATEST" ] && [ -n "${REQUIRE_RESUME:-}" ]; then
    echo "REQUIRE_RESUME set but no checkpoint in $CKPT_DIR"; exit 1
fi
echo "RUN=$RUN fact=$FACT_CKPT resume=${LATEST:-none} EXTRA=${EXTRA:-}"
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

srun torchrun --nnodes="$SLURM_JOB_NUM_NODES" --nproc_per_node=4 \
    --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    -m fact.train.stage_d_wave_decoder \
    --fact-ckpt "$FACT_CKPT" \
    --manifest "${MANIFEST:-$D/manifests/train.tsv}" \
    --dev-manifest "${DEV_MANIFEST:-$D/manifests/dev.tsv}" \
    --ckpt-dir "$CKPT_DIR" --num-workers 7 \
    $RESUME_ARG ${EXTRA:-}
