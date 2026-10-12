# Pilot Runbook — CINECA Leonardo

The pilot is the plan's week-3–6 de-risk: ~110M model
(`configs/fact_pilot.yaml`), 5K-h Emilia-EN slice, 100K steps on one node,
with the R1/R2 tripwires read at every checkpoint and the λ_ctc sweep run
first. Every command below is paste-ready; edit `--account` and the
`module load` lines in `slurm/*.sbatch` once before anything.

Gate logic: steps 0–3 are login-node/cheap-job preparation; step 4 (sweep)
picks λ_ctc; step 5 is the pilot itself; step 6 is the gate review — send
its artifacts back for the verdict-protocol review before any 2-node
stage-A arm launches.

## 0. One-time setup (login node) — if not already done

```bash
cd $WORK && git clone <repo> FaCT && cd FaCT && git checkout claude/semi-discrete-tts-representation-f0ymir
scripts/cineca_setup.sh          # base on $WORK ($SCRATCH fallback); NOTHING in $HOME;
                                 # venv + deps + Mimi/DAC/Vocos + eval set + smoke data,
                                 # all offline-ready; ends with the readiness verifier
source $FACT_BASE/fact_env.sh    # in every future shell; jobs source it via FACT_ENV
python scripts/verify_env.py --base $FACT_BASE --run-tests   # ENVIRONMENT READY + 91 tests
srun -p boost_usr_prod -A <account> --gres=gpu:1 --time=00:10:00 \
    python scripts/verify_env.py --base $FACT_BASE --expect-gpu   # prove it OFFLINE too
```

## 1. Smoke + environment gates (one short GPU job each)

```bash
mkdir -p logs
srun -p boost_usr_prod -A <account> --gres=gpu:1 --cpus-per-task=8 --time=00:30:00 \
    scripts/smoke_test.sh $WORK/smoke            # must end: SMOKE TEST PASSED
srun -p boost_usr_prod -A <account> --gres=gpu:1 --cpus-per-task=8 --time=00:30:00 \
    python scripts/vocoder_ceiling.py --eval-dir $WORK/data/eval/librispeech_test_clean \
    --limit 50 --out results/vocoder_ceiling.json   # must print: R5 verdict: GO
sbatch slurm/eval_baselines.sbatch               # Mimi/DAC numbers in published bands
```

**STOP if:** smoke fails, R5 says NO-GO, or baseline numbers fall outside
the bands in docs/CINECA.md Step 1. Fix before burning pilot compute.

## 2. Pilot data: 5K-h Emilia-EN slice (login node + one CPU-heavy job)

```bash
source $FACT_BASE/fact_env.sh    # FACT_DATA = bulk base (put it on $SCRATCH:
                                 # cineca_setup.sh $WORK/fact $SCRATCH/fact)
# Emilia download/extraction (login node; ~125 GB raw for the pilot slice):
#   full-download route (fits $SCRATCH): pull Emilia/EN tars per docs/DATA.md
#   into $FACT_DATA/data/raw/emilia and extract; or budget-capped route:
#   python scripts/emilia_chunks.py stage --lang EN --dest $FACT_DATA/data/raw/emilia --chunk-gb 150
python -m fact.data.manifest emilia --root $FACT_DATA/data/raw/emilia --out $FACT_DATA/manifests/emilia_en.tsv
shuf --random-source=<(yes 42) $FACT_DATA/manifests/emilia_en.tsv | head -n 2000000 > $FACT_DATA/manifests/pilot.tsv
#  ~2M utts x ~9 s ≈ 5K h. Reserve a held-out tail for eval:
tail -n 2000 $FACT_DATA/manifests/pilot.tsv > $FACT_DATA/manifests/pilot_dev.tsv
head -n 1998000 $FACT_DATA/manifests/pilot.tsv > $FACT_DATA/manifests/pilot_train.tsv

MANIFEST=$FACT_DATA/manifests/pilot_train.tsv OUT=$FACT_DATA/shards/pilot \
    CONFIG=configs/fact_pilot.yaml sbatch slurm/preprocess.sbatch
MANIFEST=$FACT_DATA/manifests/pilot_dev.tsv OUT=$FACT_DATA/shards/pilot_dev \
    CONFIG=configs/fact_pilot.yaml sbatch slurm/preprocess.sbatch
# Storage-tight? add EXTRA="--mel-dtype int8" to both (halves shard size).
# After preprocessing: cp $FACT_DATA/shards/pilot/stats.json $FACT_BASE/  (purge-safe copy)
```

Notes: ~180 GB of shards for 5K h — check `saldo`/quota first. The job is
resumable (resubmit after wall-time kills). Watch the final drop report;
>5% errors aborts by design. Then copy `stats.json` values into
`configs/fact_pilot.yaml` (`audio.mel_mean`/`mel_std`) and commit — the
trainer refuses mismatches.

## 3. λ_ctc sweep (R3 gate; one 1-GPU job, sequential arms)

```bash
CONFIG=configs/fact_pilot.yaml SHARDS=$FACT_DATA/shards/pilot \
    LAMBDAS=0.3,1.0,3.0 STEPS=50000 sbatch slurm/sweep_lambda_ctc.sbatch
```

Decision rule (RISK R3): largest λ whose reconstruction loss is within
noise of the smallest λ's; greedy byte-CER must fall with λ. Write the
winner into `configs/fact_pilot.yaml` `ctc.weight` and commit. If 50K-step
arms are too slow for the queue, 25K steps preserves ordering for this
decision (the sweep script notes the caveat automatically).

## 4. Pilot stage A (one node, 4 GPUs, ~100K steps)

```bash
CONFIG=configs/fact_pilot.yaml RUN=pilot sbatch --nodes=1 slurm/stage_a.sbatch
# auto-resumes from the newest checkpoint; chain with
# sbatch --nodes=1 --dependency=afterok:<jobid> slurm/stage_a.sbatch if needed
```

After each checkpoint (25K/50K/75K/100K):

```bash
CKPT=$FACT_RUNS/pilot/step_0025000.pt CONFIG=configs/fact_pilot.yaml \
    SHARDS=$FACT_DATA/shards/pilot_dev sbatch slurm/tripwires.sbatch
```

Milestone insurance (checkpoints live on $SCRATCH with the data): at each
gate, copy the checkpoint off the purge clock -
`cp $FACT_RUNS/pilot/step_0025000.pt $FACT_BASE/milestones/` (a few GB;
the only non-regenerable artifact).

Also watch training logs for: `ctc_infeasible` ~0.00 (R-fixed), `ctc`
decreasing, `recon` decreasing, `commit` n/a (FSQ), `leak_*` not collapsing
to 0 (GRL dead) or exploding.

## 5. Gate review (week-6 decision; RISK §2/§3 thresholds)

Collect and send back for the verdict-protocol review:
1. `results/tripwires/step_*.json` — flags must show:
   `r1_prosody_underused: false` (prosody perplexity ≥ ~50 of 625),
   `r1_content_leaks_f0: false` (prosody→F0 beats content→F0 by ≥5 pts),
   `r2_residual_dominant: false` (residual-zeroing delta < 3× prosody's).
2. `results/lambda_ctc_sweep/sweep_table.md` + chosen λ.
3. Final training-log line (all losses) + wall-clock per 1K steps.
4. `results/vocoder_ceiling.json`, `results/baseline_recon/summary.md`.

Outcomes: tripwires clean → green-light the three 2-node stage-A arms
(fact_base, baseline_single_fsq, baseline_vae) on the full 30–60K-h set.
R1 fired → GRL/InfoNCE strengthening pivot at pilot scale first.
R2 fired → residual β/dim pivot at pilot scale first.
Everything mediocre → stop per RISK §3; no scale-up on hope.

## Cost picture

Sweep ≈ 3 × (50K steps, 1 GPU) ≈ 1–2 days queue+run; pilot ≈ 100K steps on
4 GPUs ≈ ~1 day; tripwires/ceiling/baselines are minutes. Total well under
one GPU-week — the price of making the 10-GPU-week main runs safe.
