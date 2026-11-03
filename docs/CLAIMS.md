# Paper claims: exact wording, numbers and caveats (2026-11-03)

**How to read this.**
- Every number is on a **test set**, with a 95% paired bootstrap CI.
- Rows were selected on dev by docs/EVAL_PROTOCOL.md (rule 4).
- Sources: `results/tables/recon_{ls,emilia}.md`,
  `results/tts/test_ls_summary.json`, `results/tts/prosody_ls_summary.json`,
  `results/factorization/main_c8_rd05_ls.json`.
- Δ is FaCT minus the baseline; * means the CI excludes 0.
- These claims follow docs/AUDIT_2026-11-03.md. Wording outside these lines
  needs the same care.

**Conventions used throughout the paper.**
- **Bitrate:** "0.28 kbps tokens" is a coded bitrate. The full view is "0.28
  kbps tokens + a continuous residual of ≤ 0.14 kbps capacity", never
  "0.42 kbps".
- **Enrolment:** FaCT decodes with a speaker vector from an enrolment utterance
  (another utterance of the same speaker; side information, not in the
  bitrate). Baselines get no enrolment.
- **Latency:** worst-case algorithmic latency from `fact/eval/latency.py`:
  120 / 200 / 280 ms for the la0 / la4 / la8 decoders on the causal encoder.

## 1. TTS on FaCT tokens vs Mimi-2 tokens (headline)

**Claim.** With the same 300M LM, data, steps and selection rule, FaCT tokens
give better zero-shot TTS than Mimi tokens at a matched bitrate.

LibriSpeech test-clean, 1,237 targets of 4–10 s, 40 speakers; CIs resample
speakers.

| | FaCT LM | Mimi-2 LM | Δ |
|---|---|---|---|
| WER % | 2.56 | 8.54 | −5.98 [−6.81, −5.19]* |
| UTMOS | 3.48 | 2.53 | +0.96 [+0.79, +1.11]* |
| SIM to prompt | 0.914 | 0.827 | +0.088 [+0.072, +0.104]* |

**State alongside.**
- **Decoders differ.** FaCT tokens are decoded by the stage-D decoder with the
  prompt's speaker vector; Mimi decodes from codes only. Part of the UTMOS/SIM
  gap is therefore the decoder.
- **The WER gap also appears at codec level** (resynthesis, LibriSpeech): FaCT
  tokens-only 4.74 vs Mimi-2 8.06, Δ −3.32 [−3.71, −2.83]*.
- **Mimi config:** Mimi-2 uses 2 of Mimi's 8 codebooks to match the bitrate.
- **Size:** FaCT's LM has 3.1% more parameters, all in embeddings/heads (larger
  vocabulary).
- **Test sets:** this is one test set (LibriSpeech), and the comparison is
  against Mimi only. A FocalCodec-token LM is planned (docs/ICML_TODO.md).

## 2. Prosody control in TTS

**Claim.** Holding FaCT's content stream fixed and resampling only the prosody
stream keeps the words while giving as much pitch variation as fully free
sampling. Holding Mimi-2's first codebook keeps the words much less.

Same 1,237 targets, 3 resamples each; CIs resample speakers.

| | FaCT | Mimi-2 | Δ |
|---|---|---|---|
| words_moved (word changes vs base, fixed ÷ free) | 0.46 [0.41, 0.52] | 0.82 [0.78, 0.85] | −0.35 [−0.42, −0.29]* |
| pitch_kept (median-F0 s.d., fixed ÷ free) | 0.99 [0.94, 1.02] | 0.95 [0.91, 1.00] | +0.03 [−0.04, +0.09] (n.s.) |
| WER of content-fixed samples % | 2.58 | 9.30 | −6.72 [−7.50, −6.02]* |

**Do not claim:**
- that FaCT gives *more* pitch variation: Mimi-2's absolute diversity is
  higher;
- that pitch_kept differs between the codecs.

**State alongside.** Mimi-2's word changes sit on a noisier ASR floor (its base
WER is 8.5%), which pushes its ratio toward 1. Part of the words_moved gap is
therefore audio quality.

## 3. Codec factorization

**Claim.** The decoder takes pitch from the prosody tokens only.

LibriSpeech test-clean (main model, stage-D decoder, tokens only), WORLD-shifted
inputs:

| Pitch path | Result |
|---|---|
| ±4 st through the prosody tokens (content from the original) | output shifts +3.82 / −3.75 st |
| the same shift through the content tokens | 0.00 st (95th percentile of the absolute shift 0.02 st) |
| prosody tokens from a flattened-F0 input | intonation ratio 0.20 [0.19, 0.21] (reconstruction: 1.1) |

**Do not claim** that the content tokens are pitch-invariant: their agreement
under a pure pitch shift is only 0.39–0.41, lower than Mimi-2's first codebook
(0.64).

**Disclose** that the enrolment speaker vector carries no pitch (VC shift
0.07 st).

## 4. Non-causal intelligibility (provisional headline: ssl_nc, flow decoder)

**Claim.** At 0.28 kbps tokens + ≤ 0.14 kbps residual, the non-causal WavLM-input
FaCT has lower WER than FocalCodec 25 Hz (0.325 kbps) on LibriSpeech and ties
it on Emilia.

| ssl_nc vs FocalCodec 25 Hz | ΔWER % | ΔUTMOS | ΔSIM |
|---|---|---|---|
| LibriSpeech, full | −0.59 [−1.03, −0.23]* | −0.38* | −0.047* |
| LibriSpeech, tokens only (0.28 kbps) | −0.18 [−0.66, +0.30] (tie) | −0.34* | −0.051* |
| Emilia, full | −0.01 [−0.29, +0.28] (tie) | −0.22* | −0.018* |
| Emilia, tokens only | +0.75 [+0.47, +1.04]* | −0.16* | −0.022* |

**State alongside.**
- **Quality gap:** audio quality and speaker similarity are lower than
  FocalCodec 25 Hz.
- **Training bug:** both non-causal arms were trained while attending to batch
  padding (fixed since; tests/test_padding_invariance.py). Their numbers are
  therefore conservative.
- **Provisional:** this row is a flow-decoder row. Both non-causal encoders are
  being retrained with the padding fix (`*_pfix`, queued 2026-11-03), along
  with new stage-D decoders. The headline is re-selected on dev once they
  finish.

## 5. Causal intelligibility

**Claim (no streaming).** The causal encoder with its non-causal stage-D
decoder (dev-selected headline: disjref) matches FocalCodec 25 Hz on
LibriSpeech WER and trails on Emilia.

| disjref vs FocalCodec 25 Hz | ΔWER % | ΔUTMOS |
|---|---|---|
| LibriSpeech | +0.35 [−0.08, +0.74] (tie) | −0.67* |
| Emilia | +1.34 [+1.05, +1.63]* | −0.48* |

**Claim (streaming, la4, about 200 ms algorithmic latency).** It matches
FocalCodec-Stream on WER with tokens plus the bounded residual, at lower token
bitrate.

| la4 vs FocalCodec-Stream (0.55 kbps) | ΔWER % | ΔUTMOS | ΔSIM |
|---|---|---|---|
| LibriSpeech, full | +0.10 [−0.28, +0.42] (tie) | −0.63* | −0.019* |
| Emilia, full | −0.06 [−0.43, +0.31] (tie) | −0.27* | −0.000 |
| LibriSpeech, tokens only | +0.99 [+0.58, +1.34]* | −0.75* | −0.030* |
| Emilia, tokens only | +1.21 [+0.79, +1.65]* | −0.41* | −0.010* |

**State alongside.**
- **Latency:** FocalCodec-Stream's latency must be stated on the same
  worst-case definition; it may be lower than FaCT's.
- **Audio quality:** FocalCodec-Stream is better on UTMOS everywhere.
- **LibriTTS-trained decoders:** they narrow the UTMOS gap (la4 −0.25 on
  LibriSpeech, la8 −0.03 on Emilia) but are not the dev-selected rows. They are
  reported in the tables, not as the headline.

## 6. Low bitrate vs Mimi-2 (codec level)

**Claim.** At about 0.28 kbps, FaCT tokens reconstruct far more intelligibly
than Mimi-2 (0.275 kbps), causal decoder or not.

| FaCT tokens only vs Mimi-2 | ΔWER % | ΔUTMOS |
|---|---|---|
| main, LibriSpeech | −3.32 [−3.71, −2.83]* | +1.01* |
| main, Emilia | −4.92 [−5.32, −4.52]* | +0.90* |
| la4 (streaming), LibriSpeech | −3.41 [−3.74, −3.07]* | +0.74* |
| la4 (streaming), Emilia | −4.66 [−5.10, −4.22]* | +0.69* |

## 7. Ablations (price of causality and others)

**Example.** Non-causal vs causal with the same flow decoder at the dev-selected
setting: LibriSpeech ΔWER −7.12 [−8.03, −6.24]*, Emilia −4.93*.

**State alongside.**
- The non-causal arm was handicapped by the padding bug, so this gap is a
  conservative estimate of what non-causality buys.
- With stage-D decoders, causal models reach 3.7–4.7 WER on LibriSpeech,
  which is what the streaming claims use.

## 8. Mandatory disclosures (limitations section)

- **Enrolment:** FaCT's speaker vector comes from a second utterance (usually
  the same LibriSpeech chapter); it is not counted in the bitrate.
- **Residual:** an information bound (≤ 0.14 kbps), not a coded bitstream.
- **Padding bug** in the non-causal arms' training.
- **Emilia split:** recording-group-disjoint, not speaker-disjoint. Emilia
  transcripts are themselves ASR output.
- **Speaker similarity** uses WavLM-base-plus-SV, not the WavLM-large/ECAPA of
  most papers, so absolute SIM is not comparable with published numbers.
- **Emilia mel-L1 / energy r** are not comparable for 16 kHz codecs (shown
  n/c).
- **Audio quality:** below FocalCodec on UTMOS and SIM in every comparison.
