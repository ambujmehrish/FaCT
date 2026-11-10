# ICML: status at the freeze (2026-11-10) and what is left after CVPR

**Decision (user, 2026-11-10):** freeze the work as of today and continue
after the CVPR deadline. GPUs go to CVPR until then. Queue nothing for FaCT
without approval. The freeze record is `results/FROZEN.md`; the claims and
their exact wording are in `docs/CLAIMS.md`.

## Final at the freeze (test sets, paired 95% CIs)

1. **TTS on FaCT tokens vs Mimi-2 tokens:**
   - WER −5.98*, UTMOS +0.96*, SIM +0.088* (CIs resample speakers);
   - the headline result.
2. **Prosody control:** words_moved 0.46 vs 0.82 (Δ −0.35*).
3. **Factorization:**
   - pitch follows the prosody tokens (±3.8 st);
   - content tokens give 0.00 st.
4. **Low bitrate vs Mimi-2 at about 0.28 kbps:** WER −3.3* (LibriSpeech),
   −4.9* (Emilia).
5. **Streaming la4 (200 ms) vs FocalCodec-Stream:** WER ties on LibriSpeech
   and Emilia.
6. **Causal disjref vs FocalCodec 25 Hz:** ties on LibriSpeech; behind on
   Emilia (+1.34*).
7. **Enrolment ablation:**
   - WER does not depend on the enrolment, and the WER ties hold without it;
   - speaker similarity does depend on it.
8. **Padding bug:**
   - fixed, and verified by retraining;
   - WER unchanged for both non-causal arms;
   - causal models unaffected.
9. **Non-causal headline:**
   - ssl_nc_pfix with the flow decoder, selected on dev;
   - ΔWER vs FocalCodec 25 Hz: LibriSpeech −0.40 [−0.87, +0.09], Emilia
     +0.21 [−0.06, +0.48]; both ties;
   - the claim is softened from "lower" to "matches".
10. **Negative result:** stage-D (waveform) decoders underperform on
    non-causal encoders (LibriSpeech WER 18% vs 3.6% with the flow decoder).

## Open at the freeze (no GPUs needed unless noted)

- **Price-of-causality ablation (claim 7 in CLAIMS):** recompute with the
  retrained non-causal rows (`*_pfix`) instead of the bug-trained ones. This
  is a table-only change.
- **Archives:** done on 2026-11-10 (results/FROZEN.md, "After the freeze").

## After CVPR (each needs approval; ordered by value for reviewers)

1. **FACodec baseline** (NaturalSpeech 3; the closest factorized codec):
   - reconstruction plus the factorization (pitch-shift / swap) test;
   - the wrapper in `fact/baselines/` is a stub and needs implementing;
   - evals only, about a day.
2. **FocalCodec-token TTS model:**
   - tokenize Emilia with FocalCodec and train the same 300M LM, so the TTS
     comparison includes the strongest codec, not only Mimi-2;
   - about 2 days.
3. **Human listening study:**
   - three tests (docs/LISTENING_STUDY.md, draft);
   - needs IRB approval, the shareable Gradio app, and stimuli from the
     frozen models.
4. **Streaming demo:** a static page plus a screen recording, with la4 and the
   prosody slider (docs/DEMO_PLAN.md).
5. **Optional:**
   - quantize the enrolment vector (4/8 bits) and report its cost;
   - a larger, longer-trained generator to close the UTMOS gap to FocalCodec;
   - a speaker-disjoint Emilia split.

## Writing

- main results table with CIs (results/tables/recon_{ls,emilia}.md and the
  paired sections);
- bitrate accounting (tokens + residual bound);
- latency table (fact/eval/latency.py);
- ablation table;
- limitations, per CLAIMS section 9.
