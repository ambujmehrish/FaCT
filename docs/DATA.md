# Data setup

The pipeline is: **acquire → manifest → preprocess (shards) → stats → train**.
Preprocessing is CPU-bound; do it once per corpus and reuse the shards for
every run (FaCT, both matched baselines, and the 6.25 Hz variants all read
the same shards).

## 1. Acquire corpora (all open)

| Corpus | Role in the plan | Where |
|---|---|---|
| Emilia (~101K h, 6 langs) | main training pool; take a 5K-h slice first (week 1-2), 30-60K h for stage A | HF `amphion/Emilia-Dataset` (gated - accept terms) |
| LibriTTS-R (585 h, clean) | clean training/eval | OpenSLR 141 / HF `mythicinfinity/libritts_r` |
| Expresso | expressive eval | HF `ylacombe/expresso` |
| ESD | emotion eval (emotion-SIM) | https://hltsingapore.github.io/ESD/ |

Emilia (needs `huggingface-cli login` after accepting the gate):

```bash
huggingface-cli download amphion/Emilia-Dataset \
    --repo-type dataset --include "Emilia/EN/*.tar" --local-dir data/emilia_raw
# extract the webdataset tars -> per-utterance .mp3 + .json (with "text"):
for t in data/emilia_raw/Emilia/EN/*.tar; do tar -xf "$t" -C data/emilia/EN; done
```

LibriTTS-R:

```bash
wget https://us.openslr.org/resources/141/train_clean_360.tar.gz
tar -xzf train_clean_360.tar.gz -C data/
```

## 2. Build manifests (TSV: path<TAB>transcript)

```bash
python -m fact.data.manifest libritts --root data/LibriTTS_R --out manifests/libritts.tsv
python -m fact.data.manifest emilia   --root data/emilia/EN  --out manifests/emilia_en.tsv
# generic audio trees / jsonl metadata also supported - see module docstring.
cat manifests/*.tsv > manifests/train.tsv
```

For a 5K-h week-1 slice, `shuf manifests/emilia_en.tsv | head -n <N>` (Emilia
averages ~9 s/utt, so ~2M utterances ≈ 5K h; check `stats.json` afterwards).

## 3. Preprocess to shards

```bash
pip install -e ".[audio]"   # pyworld (F0), soundfile; torchaudio for mp3/resample
python -m fact.data.preprocess \
    --manifest manifests/train.tsv --out shards/train \
    --config configs/fact_base.yaml --workers 32 \
    --min-sec 1.0 --max-sec 30.0
```

- **Parallel:** `--workers N` (near-linear; mel+F0 dominate).
- **Resumable:** shard files are written atomically and existing shards are
  skipped - just re-run the same command after an interruption.
- **F0:** pyworld (DIO+StoneMask) when installed; otherwise a coarse
  autocorrelation fallback with a warning. Use pyworld for paper runs.
- Failed/filtered utterances are dropped and reported, never fatal.

Keep a held-out manifest (e.g. a speaker-disjoint 1%) and shard it to
`shards/dev` for the modelability probe's held-out NLL.

## 4. Set mel normalization from stats

Preprocessing writes `shards/train/stats.json`:

```json
{"n_utterances": ..., "hours": ..., "mel_mean": -4.37, "mel_std": 2.91}
```

Copy `mel_mean`/`mel_std` into `audio:` in your config (defaults are -4.0/3.0).
Flow matching wants ~unit-scale targets; this is the one config field that is
data-dependent. Recompute anytime with `--stats-only`.

## 5. Train (identical data across all fairness arms)

```bash
python -m fact.train.stage_a --config configs/fact_base.yaml            --shards shards/train --ckpt-dir runs/fact
python -m fact.train.stage_a --config configs/baseline_single_fsq.yaml  --shards shards/train --ckpt-dir runs/single_fsq
python -m fact.train.stage_a --config configs/baseline_vae.yaml         --shards shards/train --ckpt-dir runs/vae
```

RQ4 (token-count-fixed training): add `--crop-tokens 200` (vs seconds-fixed
default) with `configs/fact_6hz.yaml`.

## Not yet wired (deliberately)

- **Per-phone duration / voice-quality prosody targets** need a forced
  aligner (MFA) - the F0/energy targets are sufficient for the week 1-6
  de-risking; add aligner outputs as extra shard keys later.
- **External baselines** (Mimi etc.) need no preprocessing here: wrap their
  public checkpoints (see `fact/baselines/mimi.py`) and feed the same eval
  audio to the modelability probe.
