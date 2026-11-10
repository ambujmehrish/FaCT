# Results freeze: 2026-11-10

Work paused until after the CVPR deadline (user decision). The freeze is the
merge of this branch into `main` on 2026-11-10 (PR "Freeze 2026-11-10"). The
tag `fact-freeze-2026-11-10` could not be pushed from the sandbox; create it on
that merge commit with
`git tag -a fact-freeze-2026-11-10 <commit> -m "results freeze" && git push origin fact-freeze-2026-11-10`. After the freeze, a number changes
only through a new, documented run, recorded below under "After the freeze".

## How to read the numbers

- **Rules:**
  - the protocol is docs/EVAL_PROTOCOL.md: dev selects, test reports;
  - claim wording and caveats are in docs/CLAIMS.md;
  - remaining work is in docs/ICML_TODO.md.
- **Statistics:** paired 95% bootstrap CIs on the same utterances; * means
  the CI excludes 0.
- **Provenance:** every result JSON written since 2026-11-03 carries a
  `provenance` block (git commit, host, Slurm job, argv). Job logs are in
  results/logs/.

## Tables (rebuild with `bash scripts/collect_results.sh`)

| Table | Source files |
|---|---|
| results/tables/recon_ls.md (LibriSpeech test-clean, 2,620 utt.) | results/baseline_recon_ls/*.json, results/fact_recon/*_ls_spkother.json |
| results/tables/recon_emilia.md (Emilia test, 3,783 utt.) | results/baseline_recon_emilia/*.json |
| results/tables/recon_dev.md (dev, selection only) | results/baseline_recon_dev/*.json |
| results/tables/tripwires.md | results/tripwires/*.json |
| results/tables/{dev_sweep,wer_sweep}.md (dev, selection only) | results/fact_recon, results/baseline_recon_dev |

## Other result sets

| Result | Files |
|---|---|
| TTS, FaCT vs Mimi-2 LM | results/tts/test_ls_summary.json (speaker bootstrap), results/tts/test_ls_*.json |
| Prosody control in TTS | results/tts/prosody_ls_summary.json |
| Factorization | results/factorization/main_c8_rd05_ls.json |
| Enrolment ablation | results/enrolment_ablation/summary_paired.md, results/enrolment_ablation/{ls,emilia}/ |
| Padding diagnostic | results/padding_diagnostic/*.json |
| Excluded pilot results | results/_pilot/ (README) |

## Headline rows (selected on dev, EVAL_PROTOCOL rule 4)

| Category | Row |
|---|---|
| Causal | factwave_main_c8_rd05_disjref@0250000 (Emilia-trained stage-D decoder) |
| Streaming | factwave_main_c8_rd05_la4@0250000 (about 200 ms algorithmic latency) |
| Non-causal | fact_ssl_nc_c8_rd05_pfix, flow decoder (T=0.3, CFG=2.0), padding-fix retrain |

## Checkpoints

- **2026-11-03 archive:** 33 final checkpoints at
  `/leonardo_work/AIFAC_S07_041/fact/archive/2026-11-03`, with a sha256
  manifest, pip freeze and HF revisions (completed).
- **2026-11-10 archive:** the padding-fix retrains and their stage-D decoders
  go to `/leonardo_work/AIFAC_S07_041/fact/archive/2026-11-10` (job
  `2026-11-10-archive-checkpoints`, queued at the freeze).
- **Rebuttal backup (queued 2026-11-10, job `2026-11-10-backup-scratch`):**
  - copies everything still on SCRATCH that a rebuttal may need into
    `/leonardo_work/AIFAC_S07_041/fact/rebuttal_backup/2026-11-10`;
  - contents: 62 full checkpoints with optimizer state
    (ops/archive/rebuttal_checkpoints.txt), the Emilia dev/test manifests with
    their audio (rewritten manifests in `manifests/`), the dev shards, and the
    model caches if they are on SCRATCH;
  - `backup_manifest.json` lists every copy with sha256, what was already
    safe on WORK, and anything missing;
  - a path maps back by replacing `/leonardo_scratch` with `<backup>/scratch`.
- **Not backed up:** the training shards (`shards/train_v2`, 46,311 h of mels)
  and the TTS token shards are too large. They are rebuilt from Emilia with
  `slurm/preprocess.sbatch` and `slurm/tts_tokenize.sbatch`.
- **SCRATCH:** originals are on
  `/leonardo_scratch/large/userexternal/amehrish/fact/runs`, which is purged
  after about 40 days.

## Open at the freeze

- **Cost-of-causality ablation:** recompute from the `*_pfix` rows
  (table-only).
- **Done the same afternoon:** the ssl_nc_pfix Emilia rows and both archive
  jobs (see "After the freeze").

## Cluster state at the freeze

- **FaCT jobs:** at the freeze, only the 2 ssl_nc_pfix Emilia evals were
  running. By the afternoon of 2026-11-10 the FaCT queue was empty. Everything
  else had finished or was cancelled; the ssl_nc_pfix stage-D decoders were
  cancelled because stage-D underperforms on non-causal encoders.
- **Runner:** tmux session `fact_runner` on login01 (ops/README.md). Leave it
  running until the archive job and the 2 evals have reported, then stop it
  with `tmux kill-session -t fact_runner` if the GPUs or login node are needed.

## After the freeze

(Record any change to a number here: date, what changed, which run.)

- **2026-11-10, afternoon: ssl_nc_pfix Emilia test rows landed.** These are
  the 2 evals that were running at the freeze. Claim 4 Emilia rows filled:
  - full ΔWER vs FocalCodec 25 Hz +0.21 [−0.06, +0.48] (tie);
  - tokens only +1.26*.
  Tables rebuilt.
- **2026-11-10: rebuttal backup done** (job 64051021): 7,069 files,
  181.7 GB, to `/leonardo_work/AIFAC_S07_041/fact/rebuttal_backup/2026-11-10`.
  - The job reads FAILED only because the backup script exits non-zero when
    a listed checkpoint is missing.
  - The 6 missing checkpoints are not needed:
    - `tts_{fact,mimi2}[_b1]`: earlier TTS chains, cancelled on 2026-10-28;
      the reported TTS models are the `_b1r` ones, which were copied;
    - `wavedec_{noncausal,ssl_nc}_c8_rd05_ltts`: excluded segment-trained
      decoders.
  - The model cache was already on WORK.
- **2026-11-10: retrain-checkpoint archive done** (4 checkpoints,
  `archive/2026-11-10`).
