# Running on CINECA (Leonardo, 4x A100-64GB per node; training uses 2 nodes = 8 GPUs)

Two hard constraints shape everything here:
1. **Compute nodes have no internet** - all downloads (checkpoints, eval
   sets, pip installs) happen on a login node, into `$WORK`.
2. **Everything heavy runs inside SLURM jobs** - model inference and
   training on GPU; the preprocessing job uses its allocation's cores with
   `OMP_NUM_THREADS=1` per worker so nodes are never oversubscribed. The
   eval harness caps CPU threads via `--threads $SLURM_CPUS_PER_TASK`.

## One-time setup (login node)

One script does everything, and **nothing touches $HOME** - not the venv,
not pip/HF/torch/triton caches, not model weights, not datasets:

```bash
cd $WORK && git clone <this repo> FaCT && cd FaCT     # code on $WORK, not $HOME
scripts/cineca_setup.sh                                # or: scripts/cineca_setup.sh /path/base
```

The script picks the base dir ($WORK/fact, falling back to **$SCRATCH/fact**
when $WORK has < 400 GB free - with a warning, since Leonardo purges
$SCRATCH periodically), refuses $HOME outright, creates the venv, installs
deps (CUDA wheels via `CUDA_WHEELS=cu121`), redirects every cache
(HF_HOME, XDG_CACHE_HOME/pip, TORCH_HOME, TRITON_CACHE_DIR, ...) through a
generated **`$FACT_BASE/fact_env.sh`**, downloads Mimi/DAC/Vocos
(+ `WHISPER=large-v3` optionally; DAC's hardcoded `~/.cache/descript`
write is redirected too), stages the eval set and smoke data, and finishes
with the offline-readiness verifier.

Every job sources that env file (`FACT_ENV=.../fact_env.sh sbatch ...`;
default `$WORK/fact/fact_env.sh`).

**Verify - on the login node AND once inside a compute job** (offline,
where a missing cache would otherwise cost a queued allocation):

```bash
source $FACT_BASE/fact_env.sh
python scripts/verify_env.py --base $FACT_BASE                      # login node
srun -p boost_usr_prod -A <account> --gres=gpu:1 --time=00:10:00 \
    python scripts/verify_env.py --base $FACT_BASE --expect-gpu     # compute node
```

Both must print `ENVIRONMENT READY`: the verifier checks every cache path
is outside $HOME (and flags past leaks in `~/.cache`), free space, all
imports, model snapshots loadable with `HF_HUB_OFFLINE=1`, DAC weights,
eval/smoke data presence, and (with `--expect-gpu`) a CUDA op.

## Step 0 - real-data smoke test (run this before anything else)

```bash
DOWNLOAD_ONLY=1 scripts/smoke_test.sh $WORK/smoke     # login node: stage data
srun -p boost_usr_prod -A <account> --gres=gpu:1 --cpus-per-task=8 --time=00:30:00 \
    scripts/smoke_test.sh $WORK/smoke                 # compute node: full run
```

Exercises the exact experiment pipeline on real LibriSpeech utterances - no
synthetic data: manifest -> pyworld preprocessing (fingerprinted shards) ->
stats-derived config -> stage A -> stage B -> tokenize/detokenize/CTC
feasibility/probes. It must end with `SMOKE TEST PASSED` before any real
job is submitted. Re-runs skip the download phase automatically, so the
compute-node run is offline-safe.

## Step 1 - validate the environment with public checkpoints

```bash
mkdir -p logs && sbatch slurm/eval_baselines.sbatch    # edit --account first
```

Produces `results/baseline_recon/{mimi,dac}.json` + `summary.md`.
**Validation checks** (before trusting any of our own numbers):

- `metadata` in each JSON is read from the checkpoints, not hardcoded:
  Mimi should report 12.5 fps / 24 kHz / 8x2048 books (~1.1 kbps);
  DAC-24k reports its own hop/codebooks - the JSON is the ground truth.
- **Ordering:** DAC (high-bitrate, fidelity-first) must beat Mimi
  (~1.1 kbps) on PESQ/STOI by a wide margin. If not, the harness is broken.
- **Magnitudes:** DAC-24k PESQ-WB typically ~3.5-4.2 on clean read speech;
  Mimi @ 8 books typically ~2-2.5. Numbers far outside these bands mean a
  resampling/alignment bug, not a research finding.
- **Identity check** runs in CI (`tests/test_metrics_audio.py`): PESQ of a
  signal against itself = 4.64, STOI = 1.0 - pinning our metric wiring to
  the reference implementations.

Once these pass, the same harness evaluates FaCT checkpoints - any
fidelity claim then rests on metrics that reproduced known baseline
behavior in this exact environment.

## Step 2 - preprocess shards

```bash
# manifests on the login node (data lives on $WORK):
python -m fact.data.manifest libritts --root $WORK/data/LibriTTS_R --out $WORK/manifests/libritts.tsv
sbatch slurm/preprocess.sbatch
```

Resumable: resubmit the same job after a wall-time kill and it skips
finished shards. Copy `stats.json`'s mel_mean/mel_std into your config.

## Step 3 - training (2 nodes x 4 GPUs = 8-GPU DDP)

```bash
sbatch slurm/stage_a.sbatch                                        # FaCT
CONFIG=configs/baseline_single_fsq.yaml RUN=single_fsq sbatch slurm/stage_a.sbatch
CONFIG=configs/baseline_vae.yaml        RUN=vae        sbatch slurm/stage_a.sbatch
INIT=$WORK/runs/fact_base/step_0400000.pt sbatch slurm/stage_b_shortcut.sbatch
```

- `srun` launches one torchrun per node; c10d rendezvous
  (`--rdzv_endpoint` on the first node, port derived from the job id)
  joins them into a single 8-process world. Only global rank 0
  logs/checkpoints. This launcher path is CI-validated with two separate
  torchrun launchers joining one world.
- 8 GPUs restores the plan's compute budget; the 64 GB (vs 80 GB) cards
  are the only delta - if per-GPU batch 8 OOMs, drop to 6 and set
  `train.grad_accum: 2` to keep the effective batch.
- Wall-time chaining: the sbatch auto-resumes from the newest checkpoint;
  submit with `--dependency=afterok:<jobid>` to queue continuation.
- Stage B (decoder-only) defaults to one node; raise `--nodes` if needed.

## Notes

- Module names (`python/3.11`, `cuda`) are placeholders - check
  `module avail` on Leonardo and pin exact versions in the sbatch files.
- `--account` must be set to your CINECA project in every sbatch file.
- Keep shards, HF cache, runs, and eval sets on `$WORK` (project quota),
  not `$HOME`.
