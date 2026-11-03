# Sourced by every job script after the environment: records which code runs.
# The runner rebases its checkout every few minutes, so the commit at job start
# (not at submission) is what this job executes.
_repo="${SLURM_SUBMIT_DIR:-$PWD}"
echo "[provenance] job=${SLURM_JOB_ID:-none} host=$(hostname) start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[provenance] git_commit=$(git -C "$_repo" rev-parse HEAD 2>/dev/null || echo unavailable) dirty_tracked_files=$(git -C "$_repo" status --porcelain --untracked-files=no 2>/dev/null | wc -l)"
echo "[provenance] python=$(command -v python) torch=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo none)"
unset _repo
