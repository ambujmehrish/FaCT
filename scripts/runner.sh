#!/bin/bash
# Git-driven job queue runner - a Leonardo LOGIN node, inside tmux (see ops/README.md):
#
#   tmux new -s ops
#   cd /leonardo_work/AIFAC_S07_041/FaCT && bash scripts/runner.sh    # Ctrl-b d to detach
#
# Every INTERVAL seconds: fast-forward the checkout to origin/$BRANCH by
# rebase (never merge), then one queue tick (fact/ops/queue.py): submit new
# requests from ops/queue/, poll jobs, run post steps, push results and
# results/ops/status.json through scripts/ops_collect.sh.
# State lives outside the repo ($FACT_BASE/ops_state/state.json), so a
# restart never re-submits anything.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
BRANCH=${OPS_BRANCH:-claude/semi-discrete-tts-representation-f0ymir}
INTERVAL=${INTERVAL:-300}

# Project directories (FACT_BASE/DATA/RUNS/CACHE) + the venv.
# shellcheck disable=SC1090
source "${FACT_ENV:-$WORK/fact/fact_env.sh}" || { echo "cannot source fact_env.sh"; exit 1; }
: "${FACT_BASE:?fact_env.sh did not set FACT_BASE}"
export FACT_LOGS=${FACT_LOGS:-$REPO/logs}      # where the slurm/*.sbatch files log
export OPS_BRANCH=$BRANCH
STATE=${OPS_STATE:-$FACT_BASE/ops_state/state.json}
LOG=$FACT_LOGS/ops_runner.log
mkdir -p "$(dirname "$STATE")" "$FACT_LOGS"

cur=$(git rev-parse --abbrev-ref HEAD)
if [ "$cur" != "$BRANCH" ]; then
    echo "checkout is on '$cur', not '$BRANCH' - see ops/README.md (first-time setup)"; exit 1
fi
echo "runner: repo=$REPO branch=$BRANCH every ${INTERVAL}s state=$STATE log=$LOG"

while true; do
    # Jobs write into tracked results/ files; commit those locally first so
    # the rebase can run (they are pushed by the next collector call).
    if [ -n "$(git status --porcelain --untracked-files=no -- results)" ]; then
        bash scripts/ops_collect.sh --commit-only "local results (runner)" >> "$LOG" 2>&1
    fi
    if git fetch -q origin "$BRANCH" && git rebase -q FETCH_HEAD; then
        python -m fact.ops.queue tick --repo "$REPO" --state "$STATE" >> "$LOG" 2>&1 \
            || echo "$(date -Is) tick exited with an error" >> "$LOG"
    else
        git rebase --abort 2>/dev/null
        echo "$(date -Is) git update failed" >> "$LOG"
    fi
    tail -n 3 "$LOG"
    sleep "$INTERVAL"
done
