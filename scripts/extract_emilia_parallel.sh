#!/usr/bin/env bash
# Parallel Emilia tar extraction with a live progress line.
#
# Lustre extraction is metadata-bound (10-20K tiny files per tar): serial
# tar manages ~2-3 min/tar, parallel streams scale near-linearly. Atomic
# per-tar (.extracting -> mv), deletes each tar on success, fully
# resumable - rerun anytime, finished dirs are skipped.
#
# Usage:
#   scripts/extract_emilia_parallel.sh [P] [LANG]       # defaults: P=8, EN
# Login node: P=8 is polite. In a cheap CPU allocation, P=32:
#   srun -p lrd_all_serial -A <ACCOUNT> -n1 --cpus-per-task=4 --mem=16G \
#        --time=06:00:00 bash -lc \
#        'source $WORK/fact/fact_env.sh && $WORK/FaCT/scripts/extract_emilia_parallel.sh 32'
set -euo pipefail

P=${1:-8}
LANG_CODE=${2:-EN}
: "${FACT_DATA:?source \$FACT_BASE/fact_env.sh first}"
export SRC="$FACT_DATA/data/raw/emilia_tars/Emilia/$LANG_CODE"
export DST="$FACT_DATA/data/raw/emilia/$LANG_CODE"
mkdir -p "$DST"

# Workers rename .extracting dirs and delete tars while these run, so find
# can hit vanished entries: ignore those errors rather than abort.
count_done() {
    { find "$DST" -maxdepth 1 -type d -name "${LANG_CODE}-*" ! -name "*.extracting" 2>/dev/null || true; } | wc -l
}
count_tars() {
    { find "$SRC" -maxdepth 1 -name "*.tar" 2>/dev/null || true; } | wc -l
}

start_done=$(count_done)
start_tars=$(count_tars)
total=$((start_done + start_tars))
echo "extracting with P=$P: $start_done already done, $start_tars tars to go (total $total)"
[ "$start_tars" -eq 0 ] && { echo "nothing to extract"; exit 0; }

# Progress line, refreshed in the background while xargs works.
(
    set +e  # the monitor must never die on a transient error
    while sleep 15; do
        d=$(count_done); l=$(count_tars)
        pct=$((100 * d / total))
        printf "\r[%s] %d/%d extracted (%d%%), %d tars left    " \
               "$(date +%H:%M:%S)" "$d" "$total" "$pct" "$l"
    done
) &
MON=$!
trap 'kill $MON 2>/dev/null || true; echo' EXIT

cd "$SRC"
ls ./*.tar 2>/dev/null | xargs -r -P "$P" -I{} bash -c '
    t="{}"; stem=$(basename "$t" .tar); d="$DST/$stem"
    [ -d "$d" ] && { rm -f "$SRC/$t"; exit 0; }
    tmp="$d.extracting"; rm -rf "$tmp"; mkdir -p "$tmp"
    tar -xf "$SRC/$t" -C "$tmp" && mv "$tmp" "$d" && rm "$SRC/$t"
'

kill $MON 2>/dev/null || true
echo
echo "EXTRACTION COMPLETE: $(count_done)/$total dirs in $DST"
echo "cleanup: rm -rf \"$FACT_DATA/data/raw/emilia_tars/.cache\""
