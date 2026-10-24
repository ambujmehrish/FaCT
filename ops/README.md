# Git-driven job queue

Claude works in a sandbox without cluster access. It submits Slurm jobs on
Leonardo by committing request files, and reads results back through git.
A runner on a login node (in tmux) does the cluster side.

```
Claude: ops/queue/<id>.json --commit/push--> branch --git fetch/rebase--> runner (login node)
                                                                           | sbatch / sacct / squeue
Claude <-- results/ (+ results/ops/status.json) <-- scripts/ops_collect.sh <-'
```

- **Branch:** `claude/semi-discrete-tts-representation-f0ymir`, used by both
  Claude and the runner. Override it with `OPS_BRANCH`.
- **Account:** `AIFAC_S07_041`, fixed in `fact/ops/queue.py`. It is never
  taken from a request.
- **Slurm scripts:** `slurm/*.sbatch`, run with the repo as working directory.

## Start / stop the runner

First-time setup, so the cluster checkout tracks the queue branch:
```bash
cd /leonardo_work/AIFAC_S07_041/FaCT
git status --short | grep -v '^??'          # must print nothing (no local edits)
git fetch origin claude/semi-discrete-tts-representation-f0ymir
git checkout -B claude/semi-discrete-tts-representation-f0ymir origin/claude/semi-discrete-tts-representation-f0ymir
```
If the checkout refuses because untracked `results/` files would be
overwritten, set them aside and restore the extra ones:
```bash
mv results $FACT_DATA/results_bak_$(date +%m%d%H%M)
```
Then repeat the checkout, then run `cp -rn $FACT_DATA/results_bak_*/. results/`.

Start:
```bash
tmux new -s ops
cd /leonardo_work/AIFAC_S07_041/FaCT && bash scripts/runner.sh     # Ctrl-b d to detach
```
- Look: `tmux attach -t ops`, or `tail -f logs/ops_runner.log`.
- Stop: `tmux attach -t ops`, then Ctrl-c. Or `tmux kill-session -t ops`.
- Interval: `INTERVAL=300` by default.
- Lock: a second runner on the same state file does nothing, because of the
  lock on `<state>.lock`.

Each loop runs `git fetch origin <branch> && git rebase FETCH_HEAD` (never a
merge; on failure `git rebase --abort` and "git update failed" in the log),
then one tick. Keep the checkout free of local edits, or the rebase fails.

## Where things live

| what | where |
|---|---|
| requests | `ops/queue/<id>.json` (committed by Claude) |
| runner state (job ids, what was submitted / collected) | `$FACT_BASE/ops_state/state.json`, outside the repo, so restarts never re-submit |
| status of the queue and of ALL your jobs | `$FACT_BASE/results/ops/status.json` -> pushed as `results/ops/status.json` |
| runner log | `logs/ops_runner.log` (also pushed as `results/logs/ops_runner.txt`) |
| job logs | `logs/<job-name>-<id>.out` -> last 400 lines pushed as `results/logs/<name>.txt` (last 7 days) |
| results | the repo's `results/` (where FaCT jobs write) and `$FACT_BASE/results/` (rsynced in); json/txt/md/csv < 20 MB only |

Each tick also writes `status.json`, containing:
- for every request: job id, states, submitted / finished / collected times,
  any submit, post or invalid error, waiting_for, and the post output tail;
- the queue's request ids;
- a cluster snapshot: `squeue --me` for every job of yours (including jobs
  submitted by hand, with their pending reason), counts by state, and jobs
  that ended in the last 24 h (`sacct`).

About once an hour a heartbeat push ("ops heartbeat (status only)") sends it
to git even when no job has finished.

## Request format

One request per file, `ops/queue/<id>.json`; `id` must equal the file stem.

Submit:
```json
{"id": "2026-10-25-wer-ft-distill1", "what": "dev WER of ft_c8_rd05_distill1 @100K",
 "script": "eval_baselines.sbatch",
 "env": {"BASELINES": "fact:ckpt=${FACT_RUNS}/ft_c8_rd05_distill1/step_0100000.pt:view=full:label=x",
         "EVAL_MANIFEST": "${FACT_DATA}/manifests/dev.tsv",
         "OUT": "results/baseline_recon_dev", "EXTRA": "--wer-model large-v3 --utmos --partner-subset"},
 "sbatch": {"time": "12:00:00"},
 "after": ["2026-10-25-other-request"],
 "post": {"module": "fact.eval.<module>", "args": ["--out", "${FACT_BASE}/results/x"]},
 "collect": true}
```
Watch-only, for jobs submitted by hand. It needs no `script`/`env`/`sbatch`/`after`:
```json
{"id": "2026-10-25-watch-nc8", "what": "#3 non-causal part 3", "after_jobs": ["61816481"], "collect": true}
```

### Rules

The runner refuses anything else, and each error names the field and the rule.
- **Keys:** only `id what script env sbatch after after_jobs post collect`.
- **`id`:** `^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$`, equal to the file stem.
- **`what`:** a string of at most 200 characters.
- **`script`:** `^[a-z0-9_]+\.sbatch$`, and it must exist in `slurm/`. No paths.
- **`env`:**
  - Names match `^[A-Z][A-Z0-9_]{0,40}$`.
  - These names are denied: PATH HOME SHELL BASH_ENV ENV IFS PYTHONPATH
    PYTHONHOME PYTHONSTARTUP PROMPT_COMMAND SBATCH_ACCOUNT SLURM_CONF USER
    LOGNAME, plus anything starting with `LD_ DYLD_ BASH_FUNC SLURM_ PYTHON`.
  - Values (and post args) match `^[A-Za-z0-9_./:+=, %@{}$-]{0,400}$`.
  - `$` may appear only as `${NAME}` or `$NAME`, with NAME one of `FACT_BASE`,
    `FACT_DATA`, `FACT_RUNS`, `FACT_CACHE`, `FACT_LOGS`. They are expanded in
    Python from the runner's environment; unset is an error.
  - env is passed through the process environment, never through a shell.
- **`sbatch`:** only `time` (`HH:MM:SS`), `qos` (`^[a-z_]{1,40}$`) and `array`
  (`^[0-9,\-]{1,40}(%[0-9]{1,3})?$`).
- **`after`:** other request ids. They become `--dependency=afterok:<ids>`,
  and the request waits until they have job ids.
- **`after_jobs`:** numeric job ids, for watch-only requests.
- **`post.module`:** `^fact\.(eval|data|analysis)\.[a-z0-9_]+$`. It runs as
  `python -m <module> <args>` from the repo when every job is terminal, with a
  30-minute timeout.
- **`collect`:** true or false. When true, the collector pushes with the
  message `ops <id>: <what> (<states>)`.

### How each tick behaves

- **Submissions:** each request is submitted at most once, and the state is
  saved right after every sbatch. At most 40 sbatch calls per tick. A failed
  submission (for example QOSMaxSubmitJobPerUserLimit) is recorded and
  retried on the next tick.
- **When a request finishes:**
  - A request is finished once every job (every array task) is in a terminal
    state: COMPLETED, FAILED, CANCELLED, TIMEOUT, OUT_OF_MEMORY, NODE_FAIL,
    PREEMPTED, BOOT_FAIL or DEADLINE. A job sacct does not know yet counts as
    not finished.
- **Dependencies:** only `afterok` is supported. Chained training parts that
  must follow a TIMEOUT (`afterany`) are still submitted by hand (e.g.
  `scripts/submit_distill_runs.sh`). Add a watch-only request to track them.

## Rules for jobs submitted through the queue

- **No shared temp files:** two jobs that write the same output must not
  share a temp file. Write `<file>.<pid>.partial`, then `os.replace`.
- **Where summaries go:** under `results/<experiment>/` (in the repo) or
  `$FACT_BASE/results/<experiment>/`, so the collector finds them.
- **Fail early:** validate every argument that can fail BEFORE loading data
  or models.

## How Claude uses it

1. Write `ops/queue/<YYYY-MM-DD>-<what>.json`.
2. Run `python -m fact.ops.queue validate --repo .`.
3. Commit and push to the branch. The runner submits it within one interval.
4. Read state with `git fetch origin <branch>` and
   `results/ops/status.json` (requests + cluster snapshot), plus the
   experiment's `results/` files. Never rely on pasted terminal output.
