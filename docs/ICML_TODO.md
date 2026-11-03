# ICML: what is left (as of 2026-11-02)

**Decision (user):** GPUs are needed for the CVPR submission now. Finish only
the experiments already queued; run the items below later. ICML is about two
months away.

## Done (test sets, paired 95% CIs; see docs/EVAL_PROTOCOL.md)

- **TTS (same 300M LM, LibriSpeech test-clean, 1,237 utterances):** FaCT tokens
  vs Mimi-2 tokens:
  - WER 2.56 vs 8.54 (Δ −5.98 [−6.61, −5.35]);
  - UTMOS +0.96 [+0.92, +0.99];
  - speaker similarity +0.088.
  - Results: `results/tts/test_ls_*`.
- **Prosody control** (`results/tts/prosody_ls_summary.json`): holding FaCT's
  content stream fixed and resampling prosody gives:
  - pitch_kept 0.99 [0.94, 1.03];
  - words_moved 0.46 [0.42, 0.51].
  - Mimi-2: 0.95 and 0.82.
- **Codec factorization** (`results/factorization/`): a ±4 st shift through the
  prosody tokens gives +3.8 / −3.7 st; through the content tokens, 0.00.
- **Codec intelligibility:**
  - non-causal ssl_nc (flow decoder): LibriSpeech WER 2.79 vs FocalCodec 25 Hz
    3.38; ties on Emilia (7.86 vs 7.87);
  - causal encoder with the la4 lookahead decoder (about 200 ms algorithmic latency): ties
    FocalCodec-Stream on WER at 0.28 kbps tokens + ≤0.14 kbps residual vs 0.55 kbps.
- **Ablations:** causal / non-causal, WavLM input, distillation, chunked
  lookahead, residual capacity and dropout, disjoint speaker reference.

## Open, in the queue now: audio quality vs FocalCodec

ΔUTMOS vs FocalCodec-Stream for la4, LibriSpeech / Emilia:

| Decoder | ΔUTMOS |
|---|---|
| Emilia-trained | −0.63 / −0.27 |
| LibriTTS-trained | −0.25 / −0.10 |

FocalCodec 25 Hz is still about 0.4 UTMOS ahead of the non-causal results.

Pending:
- main model's LibriTTS decoders (plain, la8);
- disjref Emilia decoder;
- the four whole-utterance-context decoders on the non-causal runs. They
  were cancelled on 2026-11-03 and are now trained on the retrained encoders:
  `wavedec_{ssl_nc,noncausal}_c8_rd05_pfix_ctx[_ltts]`.

Then apply the dev selection rule (EVAL_PROTOCOL rule 4) for the headline rows.
If a gap remains, next is a larger generator trained longer.

## Queued 2026-11-03 (approved)

1. **Padding diagnostic:** 1 GPU, about 1 h. Covers the old noncausal and
   ssl_nc models, with causal main as the control.
2. **Enrolment ablation:** 12 evals with `spk_ref=mean` and `spk_ref=xspk`, on
   the disjref and la4 decoders.
3. **Padding-fix retrain:**
   - `{noncausal,ssl_nc}_c8_rd05_pfix`: stage A, 400K steps, 4 chained 2-node
     parts each (about 3 days);
   - then tripwires and flow-decoder evals (dev and test);
   - then `wavedec_*_pfix_ctx[_ltts]` stage-D decoders (2 parts each) and
     their dev and test evals.
4. **Archive** of the final checkpoints to WORK.

## Next step (agreed 2026-11-03): freeze

Once everything queued above has finished:

1. Apply EVAL_PROTOCOL rule 4 on dev to pick the non-causal headline.
2. Regenerate every results table from the result files.
3. Freeze them: a git tag plus a `results/FROZEN.md` that lists each table,
   its source files and the commit.
4. After the freeze, numbers change only through a new, documented run.

Selections fixed so far (rule 4, dev):
- causal headline: disjref with its Emilia decoder;
- streaming: la4 with its Emilia decoder;
- per-model decoders: Emilia-trained everywhere except chunk2, which uses
  its LibriTTS decoder.

## To run later (recommended before submission)

1. **FACodec baseline** (NaturalSpeech 3; a public pretrained factorized
   codec). Reconstruction plus the factorization (pitch-shift / swap) test.
   Today nothing in the tables is a factorized codec. Evals only, about a day.
2. **FocalCodec-based TTS LM:** tokenize Emilia with FocalCodec and train the
   same 300M LM, so the TTS comparison includes the strongest codec, not only
   Mimi-2. About a day of training after tokenization.

## Optional

- **Small listening test** (MOS or MUSHRA, about 20 listeners): UTMOS is only a
  proxy.
- **Latency table:** algorithmic latency of each causal variant (FaCT
  120 / 200 / 280 ms for la0 / la4 / la8 from `fact/eval/latency.py`, vs FocalCodec-Stream on the same definition).

## Writing

- main results table with CIs;
- bitrate accounting;
- ablation table;
- factorization and prosody figures;
- limitations: Emilia speaker identity can recur across groups; the remaining
  UTMOS gap.
