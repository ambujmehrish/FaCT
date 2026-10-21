#!/bin/bash
# Submit every final-step evaluation that is still missing, in parallel.
# Run from the repo root on the login node; safe to rerun any time: a
# job is skipped when its output JSON exists or a job of the same name is
# already queued, and a run whose final checkpoint does not exist yet is
# only reported.
#
#   bash scripts/submit_final_evals.sh
#   RUNS="main_c8_rd05" STEP=0400000 bash scripts/submit_final_evals.sh
#
# Per run (checkpoint $D/runs/<run>/step_<STEP>.pt):
#   fact-tripwires-<run>  dev-set tripwires    -> results/tripwires/<run>_<STEP>.json
#   fact-ef-ls-<run>      LibriSpeech recon    -> results/fact_recon/<run>_ls.json
#   fact-ef-em-<run>      Emilia-test recon    -> results/fact_recon/<run>_emilia.json
# and a copy of the checkpoint to $KEEP on $WORK (SCRATCH is purged after ~40
# days; ~4 GB per run, COPY=0 skips it).
# Once: full-set vocoder ceiling -> results/vocoder_ceiling_full.json.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

D=${D:-/leonardo_scratch/large/userexternal/amehrish/fact}
KEEP=${KEEP:-/leonardo_work/AIFAC_S07_041/fact/checkpoints}
ACCOUNT=${ACCOUNT:-AIFAC_S07_041}
RUNS=${RUNS:-"fact_main main_c8_rd05 main_c8_rd03 main_c16_rd05"}
STEP=${STEP:-0400000}
LS_DIR=${LS_DIR:-/leonardo_work/AIFAC_S07_041/fact/data/eval/librispeech_test_clean_full}
EM_MANIFEST=${EM_MANIFEST:-$D/manifests/test.tsv}
DEV_SHARDS=${DEV_SHARDS:-$D/shards/dev}

for p in "$LS_DIR" "$EM_MANIFEST" "$DEV_SHARDS"; do
    [ -e "$p" ] || { echo "missing: $p"; exit 1; }
done
mkdir -p logs results/tripwires results/fact_recon
queued=$(squeue --me -h -o "%j" 2>/dev/null || true)

# submit <job-name> <output> <env...> -- <sbatch args...>
submit() {
    local name=$1 out=$2; shift 2
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
    if [ -e "$out" ]; then echo "  done     $name ($out)"; return; fi
    if grep -qx "$name" <<<"$queued"; then echo "  queued   $name"; return; fi
    local id
    id=$(env "${envs[@]}" OUT="$out" sbatch --parsable --account="$ACCOUNT" --job-name="$name" "$@")
    echo "  SUBMIT   $name -> job $id"
}

for R in $RUNS; do
    CK=$D/runs/$R/step_$STEP.pt
    echo "== $R"
    if [ ! -f "$CK" ]; then
        last=$(ls "$D/runs/$R"/step_*.pt 2>/dev/null | sort | tail -n 1 || true)
        echo "  NOT READY: no step_$STEP.pt yet (newest: ${last:-none}); rerun this script later"
        continue
    fi
    submit "fact-tripwires-$R" "results/tripwires/${R}_$STEP.json" \
        CKPT="$CK" SHARDS="$DEV_SHARDS" -- slurm/tripwires.sbatch
    submit "fact-ef-ls-$R" "results/fact_recon/${R}_ls.json" \
        CKPT="$CK" EVAL_DIR="$LS_DIR" -- --time=08:00:00 slurm/eval_fact.sbatch
    submit "fact-ef-em-$R" "results/fact_recon/${R}_emilia.json" \
        CKPT="$CK" EVAL_MANIFEST="$EM_MANIFEST" -- --time=08:00:00 slurm/eval_fact.sbatch
    # Keep the final checkpoint off SCRATCH (atomic: copy, check size, rename).
    dst=$KEEP/$R/step_$STEP.pt
    if [ "${COPY:-1}" = 0 ]; then
        echo "  (COPY=0: checkpoint not copied)"
    elif [ -f "$dst" ] && [ "$(stat -c %s "$dst")" = "$(stat -c %s "$CK")" ]; then
        echo "  kept     $dst"
    else
        mkdir -p "$KEEP/$R"
        cp "$CK" "$dst.tmp" && [ "$(stat -c %s "$dst.tmp")" = "$(stat -c %s "$CK")" ] \
            && mv "$dst.tmp" "$dst" && echo "  COPIED   $dst" \
            || echo "  COPY FAILED: $CK -> $dst (rerun the script)"
    fi
done

echo "== vocoder ceiling (full LibriSpeech test-clean)"
submit "fact-vocoder-full" "results/vocoder_ceiling_full.json" \
    EVAL_DIR="$LS_DIR" LIMIT=100000 -- --time=04:00:00 slurm/vocoder_ceiling.sbatch

echo
echo "Check: squeue --me -o '%.10i %.32j %.8T %.10M %.20R' | grep fact-"
echo "When no fact-(eb|ef|tripwires) job is left: bash scripts/collect_results.sh"
