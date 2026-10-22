#!/bin/bash
# Stage C: train the mel vocoder - CINECA Leonardo, 2 nodes x 4 GPUs by default
# (--nodes=1 also works; the launcher handles both).
# Independent of any FaCT checkpoint: learns FaCT's mel space -> waveform
# from real Emilia train audio. Auto-resumes from the newest checkpoint;
# chain 24 h jobs with --dependency=afterany:<jobid>.
#SBATCH --job-name=fact-vocoder-train
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
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1   # 7 audio-decoding workers per GPU use the cores

D=${FACT_DATA:-$FACT_BASE}
RUN=${RUN:-vocoder}
CKPT_DIR=${FACT_RUNS:-$D/runs}/$RUN
RESUME_ARG=""
LATEST=$(ls "$CKPT_DIR"/step_*.pt 2>/dev/null | sort | tail -1 || true)
[ -n "$LATEST" ] && RESUME_ARG="--resume $LATEST"
if [ -z "$LATEST" ] && [ -n "${REQUIRE_RESUME:-}" ]; then
    echo "REQUIRE_RESUME set but no checkpoint in $CKPT_DIR - refusing to start from scratch"; exit 1
fi
echo "RUN=$RUN resume=${LATEST:-none} EXTRA=${EXTRA:-}"
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
# Steps: 250K at 8 GPUs x 16 segments = 32M segments, the Vocos budget
# (2M steps x 16). On 1 node pass STEPS=500000 for the same budget. The
# world size and step count are recorded; a resume with others is refused.
STEPS=${STEPS:-250000}

srun torchrun \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc_per_node=4 \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    -m fact.train.stage_c_vocoder \
    --steps "$STEPS" \
    --config "${CONFIG:-configs/fact_base.yaml}" \
    --manifest "${MANIFEST:-$D/manifests/train.tsv}" \
    --dev-manifest "${DEV_MANIFEST:-$D/manifests/dev.tsv}" \
    --ckpt-dir "$CKPT_DIR" \
    --num-workers 7 \
    $RESUME_ARG ${EXTRA:-}
# Submit (~1 day on 2 nodes; the chained job is a spare that resumes or exits):
#   J=$(sbatch --parsable --account=<acct> slurm/stage_c_vocoder.sbatch)
#   sbatch --account=<acct> --dependency=afterany:$J slurm/stage_c_vocoder.sbatch
# Progress: grep -E "step [0-9]+/|val_" logs/fact-vocoder-train-<jobid>.out | tail
# Then score FaCT + vocoder with the baseline harness (identical metrics):
#   BASELINES="fact_vocoder:vocoder=<runs>/vocoder/step_XXXXXXX.pt:label=vocoder_copysyn" \
#     EVAL_DIR=<librispeech_test_clean_full> OUT=results/baseline_recon_ls \
#     EXTRA="--wer-model large-v3" sbatch --account=<acct> slurm/eval_baselines.sbatch
#   BASELINES="fact:ckpt=<run>/step_0400000.pt:vocoder=<voc.pt>:view=full:label=<run>_full" ...
