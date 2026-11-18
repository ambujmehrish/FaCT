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

0. **Audio quality (UTMOS/SIM gap to FocalCodec), diagnosed 2026-11-18:**
   - **Cause:** every audio-producing part (flow decoder, stage-C vocoder,
     Emilia stage-D decoders) is trained on noisy in-the-wild Emilia. Our
     vocoder fed ground-truth mel reaches only UTMOS 3.60 on LibriSpeech
     (2.98 Emilia), while FocalCodec scores 4.14.
   - **How FocalCodec, FocalCodec-Stream and ZipCodec differ:** they train
     their decoder on clean LibriTTS train-clean-100 for 3–5M steps; we
     train for 250K. In audio seen we are already ahead (250K steps × global
     batch 64 × 2.54 s ≈ 41M s vs about 21M s for FocalCodec), so steps are
     not the cause.
   - **Queued 2026-11-18:** phase 0 (`spk_ref=other3`: mean speaker vector
     of up to 3 other utterances; dev rows with SIM for selection, test rows
     in `results/enrolment_ablation/`) and phase 1 (`*_clean35` decoders,
     same 17.3M generator and settings, `train_dnsmos3.5.tsv`; audio read
     from SCRATCH, nothing copied). Request ids `2026-11-18-p0-*`, `-p1-*`.
   - **Phase 2 code ready, not queued:** `--spk-loss-coef` /
     `--spk-loss-model` in `fact/train/stage_d_wave_decoder.py` (pass via
     `EXTRA`); loss in `fact/vocoder/speaker_loss.py`. Default model ECAPA-TDNN
     (`speechbrain/spkrec-ecapa-voxceleb`), a different architecture from the
     evaluation SV model (WavLM-base-plus-sv, which the code refuses).
     Verified in the sandbox: embeddings equal speechbrain's `encode_batch`
     and gradients reach the decoder output. Same generator size.
     - **One-time setup on a login node** (pins torch so pip cannot move it):
       ```bash
       source /leonardo_work/AIFAC_S07_041/fact/fact_env.sh && cd /leonardo_work/AIFAC_S07_041/FaCT
       pip freeze | grep -iE '^(torch|torchaudio|numpy|scipy|huggingface[-_]hub|transformers)==' > $FACT_BASE/sb_constraints.txt
       pip install "speechbrain==1.0.3" -c $FACT_BASE/sb_constraints.txt
       python scripts/download_checkpoints.py --baselines ecapa      # about 80 MB into $FACT_CACHE/hf
       ```
     - **Plan:** after the phase 1 dev rows, one run per decoder on the same
       clean data with `--spk-loss-coef 1.0` (run names `*_clean35_spk`),
       compared with `*_clean35` on dev.
   - **Plan:** keep the tokenizer on all of Emilia. Retrain only the decoders
     (la4 and disjref stage-D, optionally the stage-C vocoder) on clean
     Emilia, using `manifests_backup/quality/train_dnsmos3.5.tsv` (1.9k h)
     or `train_dnsmos3.4.tsv` (10k h), optionally plus LibriTTS-R.
     - **Decoder size is fixed (user decision, 2026-11-18):** the same
       generator as now, dim 512 / intermediate 1536 / 8 layers, 17.3M
       parameters, the structure of FocalCodec's decoder. Only the training
       data changes, so any gain is attributable to the data.
     - same 250K steps and schedule as the current decoders;
     - add a speaker-consistency loss using a speaker-verification model
       other than the evaluation one (a training-loss change, same size);
     - report decoder parameter counts in the paper. For reference, at our
       input width: FocalCodec-Stream-sized decoder (1024/2048/8) 43.0M,
       ZipCodec-sized (1024/4096/20) 177.4M; the FaCT tokenizer is 118.7M.
   - **Cheap first step:** average the enrolment speaker vector over 3
     utterances (evaluation only).
   - **Watch:** Emilia SIM may drop with a cleaner decoder (as with the
     LibriTTS decoders).

1. **FACodec baseline** (NaturalSpeech 3; the closest factorized codec):
   - reconstruction plus the factorization (pitch-shift / swap) test;
   - the wrapper in `fact/baselines/` is a stub and needs implementing;
   - evals only, about a day.
2. **ZipCodec baseline** (arXiv 2609.11642, Sep 2026, by the FocalCodec
   authors; code, checkpoints and demo at lucadellalib.github.io/zipcodec-web):
   - **Direct competitor for the streaming claim.** It is a streaming codec at
     6.25 Hz and 0.80 kbps (64 dims × 4 levels via scalar spherical
     quantization, so a single stream, not content/prosody factorized), with
     160 ms theoretical latency. It distils WavLM layer 6 into a causal
     842M-parameter transformer, trained on 94k h, at 16 kHz. It runs in real
     time on a CPU.
   - **Published LibriSpeech test-clean numbers:**
     - ZipCodec: UTMOS 3.89, dWER 2.83, Sim 98.0;
     - FocalCodec-Stream@50 (0.80 kbps): 3.85 / 3.68 / 97.0;
     - Mimi (0.83 kbps): 3.44 / 4.77 / 96.6.
   - **Comparability:** dWER (Whisper-small, vs the original audio's
     transcript) is not comparable with our WER (Whisper large-v3 vs the
     ground truth). UTMOS is comparable: their FocalCodec-Stream UTMOS ≈ ours
     for the 0.55 kbps variant (3.88).
   - **Risk for claim 5:** ZipCodec improves on FocalCodec-Stream, which
     already beats FaCT la4 on UTMOS (3.25 Emilia decoder, about 3.6
     LibriTTS decoder, vs about 3.9). Expect ZipCodec to beat la4 on quality
     and probably on WER at 2.9× our token bitrate.
   - **Our differentiators:**
     - 0.28 kbps tokens (≤0.42 with the residual) vs 0.80;
     - factorized content/prosody tokens, with prosody control and the TTS
       results;
     - 200 ms vs 160 ms latency is a slight disadvantage;
     - 12.5 Hz vs 6.25 Hz means more LM steps for us.
   - **Plan:**
     - wrap the released checkpoint in `fact/baselines/`;
     - run LibriSpeech and Emilia reconstruction (paired vs la4, same
       Whisper WER/SIM/UTMOS);
     - also run the factorization test (pitch shift / swap) to show what its
       single stream cannot separate;
     - one GPU, a few hours.
   - **Wording:** state the streaming claim as "the lowest-bitrate streaming
     codec with factorized tokens", not "the best streaming codec".
3. **FocalCodec-token TTS model:**
   - tokenize Emilia with FocalCodec and train the same 300M LM, so the TTS
     comparison includes the strongest codec, not only Mimi-2;
   - about 2 days.
4. **Human listening study:**
   - three tests (docs/LISTENING_STUDY.md, draft);
   - needs IRB approval, the shareable Gradio app, and stimuli from the
     frozen models.
5. **Streaming demo:** a static page plus a screen recording, with la4 and the
   prosody slider (docs/DEMO_PLAN.md).
6. **U-Codec reference row** (arXiv 2510.16718; weights: hf.co/shaunxsyang/U-Codec):
   - **Not a direct competitor.** It solves a different problem: a 5 Hz,
     non-causal, non-factorized codec aimed at fewer LM steps (0.5–1.3 kbps,
     16 kHz). It has no streaming and no content/prosody split. Use it as a
     low-frame-rate reference row and in related work, not as a headline
     comparison.
   - **Plan:** wrap it in `fact/baselines/` and run the LibriSpeech and Emilia
     reconstruction evals in our harness (same Whisper WER and SIM as every
     other row). One GPU, a few hours.
   - **From its paper** (LibriSpeech test-clean; PESQ/STOI/UTMOS are
     comparable, since their Mimi-8 and BigCodec rows match ours; WER/SIM are
     not, since they use a HuBERT ASR and a WavLM-large SV):
     - 5 Hz, 16/32 codebooks: 0.96/1.28 kbps, UTMOS 3.51/3.48, PESQ-WB
       2.41/2.59, STOI 0.92/0.93;
     - FaCT ssl_nc tokens only: 0.28 kbps, UTMOS 3.80, PESQ-WB 1.24,
       STOI 0.72.
7. **Optional:**
   - quantize the enrolment vector (4/8 bits) and report its cost;
   - a larger generator: only as a separately labelled scaling ablation,
     never in the main comparison rows (decoder size is fixed);
   - a speaker-disjoint Emilia split.

## Writing

- main results table with CIs (results/tables/recon_{ls,emilia}.md and the
  paired sections);
- bitrate accounting (tokens + residual bound);
- latency table (fact/eval/latency.py);
- ablation table;
- related work: low-frame-rate codecs for LM-based TTS.
  - Offline: U-Codec (5 Hz), DualCodec, TaDiCodec, FlexiCodec.
  - Streaming: Mimi (12.5 Hz, 0.83–1.1 kbps), FocalCodec-Stream (50 Hz,
    0.55–0.80 kbps), ZipCodec (6.25 Hz, 0.80 kbps, 160 ms).
  - None factorizes content and prosody. FaCT targets streaming (200 ms),
    factorized and controllable tokens at 0.28 kbps;
- limitations, per CLAIMS section 9.
