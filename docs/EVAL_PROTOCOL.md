# Evaluation protocol: dev selects, test reports

**Rule.** Every number in the paper, and every claimed win or loss, comes from a
test set. Dev numbers are for analysis and for the selections listed below.
No setting, checkpoint, decoder or headline variant is chosen by looking at
test results.

## Sets

| role | set | where the results go |
|---|---|---|
| selection / analysis | Emilia-EN dev (`manifests/dev.tsv`, 2,000 utterances, group-disjoint from train and test) | `results/baseline_recon_dev`, `results/tables/recon_dev.md`, `wer_sweep.md`, `dev_sweep.md`, tripwires (`shards/dev`), TTS sampling sweep |
| report | LibriSpeech test-clean, all 2,620 utterances (`data/eval/librispeech_test_clean_full`) | `results/baseline_recon_ls`, `recon_ls.md`, factorization, TTS test |
| report | Emilia-EN test (`manifests/test.tsv`, group-disjoint from train and dev) | `results/baseline_recon_emilia`, `recon_emilia.md` |

The 200-utterance `librispeech_test_clean` subset (eval_baselines default) is
for smoke tests only and never reported.

## Selections, all made on dev, fixed before test results are read

1. **Flow-decoder sampling (T, CFG):** lowest dev WER with mel-L1 at most
   the reference + 0.02 (`scripts/select_wer_setting.py`). Runs without their
   own sweep use the main model's dev-selected setting (T=0.3, CFG=2.0).
2. **Checkpoints:** fixed in advance (stage A 400K, stage C 250K, stage D 250K,
   TTS 200K). No best-on-dev or best-on-test checkpoint picking.
3. **TTS sampling:** lowest dev WER; ties within 0.1 go to the higher SIM
   (`scripts/select_tts_sampling.py`).
4. **Stage-D decoder per model (Emilia- vs LibriTTS-trained) and the headline
   variant per category (causal: main / disjref / chunk2 / distill;
   non-causal: noncausal / ssl_nc):** lowest dev WER on the full view, ties
   within 0.2 go to the higher dev UTMOS. All variants are still reported in
   the test tables; the rule only picks which row is the headline.

Test-only results that are not a selection (e.g. a fixed ablation's row) are
reported as they are, wins and losses alike.
