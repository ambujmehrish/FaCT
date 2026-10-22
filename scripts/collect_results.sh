#!/bin/bash
# Build every results table and push the result files to a NEW branch
# (results/<timestamp>), never to main, so they are reviewed and merged
# through a PR. Run from the repo root on the login node when the eval and
# tripwire jobs have finished:
#
#   bash scripts/collect_results.sh            # refuses while FaCT eval jobs are running
#   FORCE=1 bash scripts/collect_results.sh    # collect whatever exists now
#
# Pushing needs GitHub credentials on the node; if the push fails the commit
# stays on the local results/<timestamp> branch (see the printed bundle fallback).
#
# Commits only small text results (JSON/Markdown under results/); checkpoints
# (*.pt) are git-ignored, and any file over MAX_MB is refused.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
MAX_MB=${MAX_MB:-20}

# 1) Do not collect half-finished results.
running=$(squeue --me -h -o "%j" 2>/dev/null | grep -E "^fact-(eb|ef|tripwires|vocoder)" || true)
if [ -n "$running" ] && [ -z "${FORCE:-}" ]; then
    echo "Eval/tripwire jobs still queued or running:"; echo "$running" | sort | uniq -c
    echo "Wait for them, or rerun with FORCE=1 to collect what exists now."; exit 1
fi

# 2) Tables.
mkdir -p results/tables
# Single-threaded and failure-tolerant: a login node may kill a table step;
# the raw JSONs are still committed and the tables can be rebuilt elsewhere.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
failed=()
if python scripts/tripwire_table.py --out results/tables/tripwires.md > /dev/null; then
    echo "tripwires -> results/tables/tripwires.md"
else failed+=(tripwires); fi

for SET in ls emilia; do
    case $SET in
        ls) TITLE="LibriSpeech test-clean (full, 2,620 utterances)" ;;
        emilia) TITLE="Emilia-EN test (5,000 utterances, speaker-disjoint)" ;;
    esac
    shopt -s nullglob
    base=(results/baseline_recon_$SET/*.json)
    facts=()
    for f in results/fact_recon/*_$SET.json; do
        label=$(basename "$f" "_$SET.json"); facts+=("$label=$f")
    done
    shopt -u nullglob
    if [ ${#base[@]} -eq 0 ] && [ ${#facts[@]} -eq 0 ]; then
        echo "$SET: no results yet, skipped"; continue
    fi
    args=(--title "$TITLE" --out "results/tables/recon_$SET.md")
    [ ${#base[@]} -gt 0 ] && args+=(--baselines "${base[@]}")
    [ ${#facts[@]} -gt 0 ] && args+=(--fact "${facts[@]}")
    if python scripts/make_recon_table.py "${args[@]}" > /dev/null; then
        echo "$SET: ${#base[@]} baselines, ${#facts[@]} FaCT runs -> results/tables/recon_$SET.md"
    else
        echo "$SET: table build FAILED (raw JSONs are still committed)"; failed+=("recon_$SET")
    fi
done

# 3) What gets committed: text results only, size-checked.
paths=(results/tables results/tripwires results/fact_recon results/baseline_recon_ls
       results/baseline_recon_emilia results/vocoder_ceiling.json results/vocoder_ceiling_full.json)
existing=()
for p in "${paths[@]}"; do if [ -e "$p" ]; then existing+=("$p"); fi; done
big=$(find "${existing[@]}" -type f -size +"${MAX_MB}"M 2>/dev/null || true)
if [ -n "$big" ]; then
    echo "Refusing: files over ${MAX_MB} MB (raise MAX_MB only if they really belong in git):"
    echo "$big"; exit 1
fi

# 4) Commit on a new branch WITHOUT touching the working tree or the current
#    branch: build the commit in a throwaway index (switching branches would
#    delete these untracked result files from the checkout when switching back).
BR="results/$(date +%Y%m%d-%H%M%S)"
export GIT_INDEX_FILE
GIT_INDEX_FILE=$(mktemp "${TMPDIR:-/tmp}/fact-results-index.XXXXXX")
trap 'rm -f "$GIT_INDEX_FILE"' EXIT
git read-tree HEAD
git add -f -- "${existing[@]}"
TREE=$(git write-tree)
if [ "$TREE" = "$(git rev-parse 'HEAD^{tree}')" ]; then
    echo "Nothing new to commit (results identical to HEAD)."; exit 0
fi
MSG="Results: $(ls results/tables 2>/dev/null | tr '\n' ' ')${failed:+ (table build failed: ${failed[*]})}

Collected by scripts/collect_results.sh on $(hostname) at $(date -Iseconds)."
COMMIT=$(git commit-tree "$TREE" -p HEAD -m "$MSG")
git branch "$BR" "$COMMIT"
echo "Committed $COMMIT on local branch $BR (working tree unchanged):"
git diff --stat HEAD "$COMMIT" | tail -n 1
if git push -u origin "$BR"; then
    echo; echo "Pushed branch $BR. Open a PR from it (or tell Claude the branch name)."
else
    echo; echo "Push failed (no GitHub credentials on this node?). The commit is on local branch $BR."
    echo "Fallback: git bundle create results.bundle HEAD..$BR  and copy it off-cluster."
    exit 1
fi
