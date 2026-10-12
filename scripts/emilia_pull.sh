#!/usr/bin/env bash
# Kill-proof Emilia downloader for CINECA login nodes.
#
# Login nodes enforce per-process CPU/memory caps and SIGKILL anything that
# exceeds them - a monolithic `hf download` of 1.2 TB dies repeatedly. This
# script makes each process too small to kill: ONE tar per hf invocation
# (~1 GB, seconds of CPU), extracted and deleted immediately. The driving
# shell loop uses no CPU; everything is resumable (extracted dirs skipped,
# completed tars reused, interrupted tars retried).
#
# Usage (login node, inside tmux, after `source $FACT_BASE/fact_env.sh`):
#   scripts/emilia_pull.sh [LANG]        # default EN
# Peak disk: extracted-so-far + one tar. Keep .json sidecars (transcripts).
set -euo pipefail

LANG_CODE=${1:-EN}
: "${FACT_DATA:?source \$FACT_BASE/fact_env.sh first}"
export HF_HUB_DISABLE_XET=1            # xet reconstruction is the CPU hog

TARS_DIR="$FACT_DATA/data/raw/emilia_tars"
OUT_DIR="$FACT_DATA/data/raw/emilia/$LANG_CODE"
LIST="$FACT_DATA/emilia_${LANG_CODE}_tars.txt"
mkdir -p "$TARS_DIR" "$OUT_DIR"

if [ ! -s "$LIST" ]; then
    python - "$LANG_CODE" "$LIST" <<'PY'
import sys
from huggingface_hub import HfApi
lang, out = sys.argv[1], sys.argv[2]
fs = sorted(f for f in HfApi().list_repo_files("amphion/Emilia-Dataset",
                                               repo_type="dataset")
            if f.endswith(".tar")
            and (f.startswith(f"Emilia/{lang}/") or f.startswith(f"{lang}/")))
assert fs, f"no tars for {lang}"
open(out, "w").write("\n".join(fs) + "\n")
print(f"{len(fs)} tars listed -> {out}")
PY
fi

total=$(wc -l < "$LIST")
n=0
while read -r f; do
    n=$((n + 1))
    stem=$(basename "$f" .tar)
    d="$OUT_DIR/$stem"
    [ -d "$d" ] && continue
    until hf download amphion/Emilia-Dataset "$f" --repo-type dataset \
          --local-dir "$TARS_DIR" > /dev/null; do
        echo "[$n/$total] retry $stem"; sleep 5
    done
    tmp="$d.extracting"
    rm -rf "$tmp" && mkdir -p "$tmp"
    tar -xf "$TARS_DIR/$f" -C "$tmp"
    mv "$tmp" "$d"                      # atomic: resume never sees partials
    rm "$TARS_DIR/$f"
    echo "[$n/$total] done $stem"
done < "$LIST"

rm -rf "$TARS_DIR/.cache"
echo "ALL $LANG_CODE TARS EXTRACTED -> $OUT_DIR"
echo "Next: python -m fact.data.manifest emilia --root $OUT_DIR --out \$FACT_DATA/manifests/emilia_${LANG_CODE,,}.tsv --require-text"
