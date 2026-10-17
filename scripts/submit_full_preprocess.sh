#!/bin/bash
# Submit one preprocessing job per train chunk (from scripts/split_emilia.py)
# into ONE shard directory, then a final --stats-only job that waits for all
# of them. Resumable: rerun the script after any failure; finished shards
# are skipped, and the stats job only runs when every chunk succeeded.
#
#   ACCOUNT=AIFAC_S07_041 CONFIG=configs/fact_base.yaml \
#     bash scripts/submit_full_preprocess.sh <manifests dir> <shards out dir>
set -euo pipefail
M=${1:?manifests dir (contains train_cNNN.tsv)}
OUT=${2:?shard output dir}
ACCOUNT=${ACCOUNT:?set ACCOUNT}
PART=${PARTITION:-boost_usr_prod}
CONFIG=${CONFIG:-configs/fact_base.yaml}
MEL_DTYPE=${MEL_DTYPE:-fp16}

shopt -s nullglob
chunks=("$M"/train_c[0-9][0-9][0-9].tsv)
[ ${#chunks[@]} -gt 0 ] || { echo "no train_cNNN.tsv in $M"; exit 1; }
ids=()
for c in "${chunks[@]}"; do
    tag=$(basename "$c" .tsv); tag=${tag#train_}          # c000, c001, ...
    jid=$(MANIFEST="$c" OUT="$OUT" CONFIG="$CONFIG" \
          EXTRA="--shard-prefix ${tag}_ --skip-stats --mel-dtype $MEL_DTYPE" \
          sbatch --parsable -A "$ACCOUNT" -p "$PART" -J "fact-pre-$tag" slurm/preprocess.sbatch)
    ids+=("$jid"); echo "$tag -> $jid"
done
dep=$(IFS=:; echo "${ids[*]}")
sjid=$(sbatch --parsable -A "$ACCOUNT" -p "$PART" -J fact-pre-stats \
       --dependency=afterok:"$dep" --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=12:00:00 \
       -o logs/%x-%j.out --wrap "source \${FACT_ENV:-\$WORK/fact/fact_env.sh}; export PYTHONUTF8=1 PYTHONUNBUFFERED=1; \
       python -m fact.data.preprocess --out $OUT --config $CONFIG --stats-only")
echo "stats job $sjid runs after all ${#ids[@]} chunks succeed"
