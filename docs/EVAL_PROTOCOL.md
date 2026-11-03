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

## Excluded runs

Excluded because of a bug, independent of their results:

- **Stage-D decoders on the non-causal encoders (noncausal, ssl_nc) trained on
  2.54 s clips** (`wavedec_{noncausal,ssl_nc}_c8_rd05[_ltts]`). A bidirectional
  encoder gives different features on a short clip than on the whole utterance
  it encodes at evaluation, so these decoders were trained on inputs they never
  see at test time.
  - Fix: `--utterance-context`, which stage D now requires for such encoders.
  - Replaced by the `wavedec_*_ctx*` runs.
  - The LibriTTS-trained versions' result files were removed (2026-11-02);
    the Emilia-trained versions were cancelled before evaluation.

## Speaker enrolment (decided 2026-11-03)

FaCT's decoder takes one global speaker vector (256-d). In every reported FaCT
row it is computed from an **enrolment utterance**: a different utterance of
the same speaker (`spk_ref=other`), never the target itself (checked in the
audit).

The paper states:
- FaCT is an enrolment-based codec. The enrolment speech is side information,
  sent once per speaker, and is not counted in the per-utterance bitrate.
- Baselines receive no enrolment.
- On LibriSpeech the enrolment utterance is usually from the same chapter (the
  same recording session) as the target.
- In TTS, the enrolment is the prompt, which both LMs receive.

Optional evidence of how much the enrolment contributes (eval only, about 1
GPU-hour):
- FaCT with a fixed average speaker vector, i.e. no enrolment;
- FaCT with the vector quantized to 4 or 8 bits, with its per-utterance cost
  reported.
