#!/bin/bash
# Results collector for the job queue (called by fact/ops/queue.py; also
# usable by hand on a login node):
#
#   bash scripts/ops_collect.sh "<message>"
#
# Default (full collect):
# 1. rsync $FACT_BASE/results/ -> repo results/ (json/txt/md/csv < 20 MB only;
#    never audio, arrays or checkpoints). Most FaCT jobs write straight into
#    the repo's results/, which step 3 picks up too.
# 2. Last 400 lines of every log in $FACT_LOGS changed in the last 7 days ->
#    results/logs/<name>.txt.
# 3. git add results/ (same filter), commit "Results: <message>" (skipped if
#    nothing changed), then fetch + rebase onto origin/$BRANCH and push.
#
#   bash scripts/ops_collect.sh --commit-only "<message>"
#      step 3 without fetch/rebase/push: the runner calls this before each
#      update, because FaCT jobs write into tracked files under results/
#      (e.g. summary.md), which would otherwise block the rebase.
set -uo pipefail
cd "$(dirname "$0")/.."
COMMIT_ONLY=""
if [ "${1:-}" = --commit-only ]; then COMMIT_ONLY=1; shift; fi
MSG=${1:?usage: ops_collect.sh [--commit-only] "<message>"}
BRANCH=${OPS_BRANCH:-claude/semi-discrete-tts-representation-f0ymir}
: "${FACT_BASE:?source fact_env.sh first}"
LOGS=${FACT_LOGS:-$PWD/logs}
mkdir -p results/logs

if [ -z "$COMMIT_ONLY" ] && [ -d "$FACT_BASE/results" ]; then
    rsync -a --prune-empty-dirs --max-size=20m \
        --include='*/' --include='*.json' --include='*.txt' --include='*.md' --include='*.csv' \
        --exclude='*' "$FACT_BASE/results/" results/ || { echo "rsync failed"; exit 1; }
fi
if [ -z "$COMMIT_ONLY" ] && [ -d "$LOGS" ]; then
    find "$LOGS" -maxdepth 1 -type f -mtime -7 -print0 | while IFS= read -r -d '' f; do
        name=$(basename "$f"); tail -n 400 "$f" > "results/logs/${name%.*}.txt"
    done
fi

find results -type f \( -name '*.json' -o -name '*.txt' -o -name '*.md' -o -name '*.csv' \) \
    -size -20M -print0 | xargs -0 -r git add -f --
if git diff --cached --quiet; then
    echo "collector: nothing changed"; exit 0
fi
git commit -q -m "Results: $MSG" || { echo "commit failed"; exit 1; }
[ -n "$COMMIT_ONLY" ] && { echo "collector: committed locally ($MSG)"; exit 0; }
if git fetch -q origin "$BRANCH" && git rebase -q FETCH_HEAD && git push -q origin "HEAD:$BRANCH"; then
    echo "collector: pushed ($MSG)"
else
    git rebase --abort 2>/dev/null
    echo "collector: push failed; the commit stays local and goes out with the next push"; exit 1
fi
