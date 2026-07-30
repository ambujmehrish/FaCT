# Running on CINECA (Leonardo, 4x A100-64GB per node; training uses 2 nodes = 8 GPUs)

Two hard constraints shape everything here:
1. **Compute nodes have no internet** - all downloads (checkpoints, eval
   sets, pip installs) happen on a login node, into `$WORK`.
2. **Everything heavy runs inside SLURM jobs** - model inference and
   training on GPU; the preprocessing job uses its allocation's cores with
   `OMP_NUM_THREADS=1` per worker so nodes are never oversubscribed. The
   eval harness caps CPU threads via `--threads $SLURM_CPUS_PER_TASK`.

## One-time setup (login node)

```bash
cd $WORK && git clone <this repo> FaCT && cd FaCT
module load python/3.11
python -m venv $WORK/fact-venv && source $WORK/fact-venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121  # match module cuda
pip install -e ".[audio,dev]" transformers descript-audio-codec pesq pystoi datasets

export HF_HOME=$WORK/hf_cache                      # put the HF cache on $WORK
python scripts/download_checkpoints.py --baselines mimi,dac   # + xcodec2/whisper later
python scripts/prepare_eval_set.py --out $WORK/data/eval/librispeech_test_clean -n 200
```

Add `export HF_HOME=$WORK/hf_cache` to your shell profile; jobs set
`HF_HUB_OFFLINE=1` so a missing download fails fast instead of hanging.

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
