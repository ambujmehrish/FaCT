# Pipeline Analysis & Scope of Change (post-literature-check)

*Companion to RISK_ASSESSMENT.md (2026-09-30 literature state). This maps
every pipeline component against the closest published work, states where
novelty actually lives now, defines the win-set we can realistically beat
baselines on, and prioritizes the changes.*

## 1. Component-by-component: where we stand vs the literature

| Component (as built) | Closest published work | Verdict on novelty | Change required |
|---|---|---|---|
| Causal conv+transformer encoder, 50→12.5 Hz | Mimi, NanoCodec (causal); SiTok (non-causal) | Commodity — never claim it | none |
| **Semi-discrete FSQ bottleneck exposing BOTH views as interface** | VoxCPM/VoxCPM2 (FSQ-as-regularization, *internal* to a tokenizer-free TTS); NeuCodec (FSQ, discrete-only interface) | **Novel at the interface level**: nobody ships a tokenizer whose continuous and discrete views address the same lattice state for downstream consumers. VoxCPM2 is the scoop risk — verify on full read. | Make the duality *operational*, not just descriptive: prosody/content edits in code-space with lattice re-rounding (P0.3), and the RQ3 hybrid interface framed as "index + intra-cell offset" (bounded correction ≤ half cell — a quantifiable property no baseline has) |
| **Prosody as a separate stream at the LM interface** | FACodec (FVQ prosody substream, non-causal, 2024); DisCodec (tri-factor but FUSES content+prosody into one LM stream); ProsoCodec (VC-oriented); Kanade (single fused stream) | **Novel as stated in RISK §1**: causal + semi-discrete + *separately addressable* prosody. The fusion step in DisCodec is our sharpest contrast — they concede the separate-stream LM interface doesn't survive their pipeline; we make it the product. | (a) Ship a prosody-edit API (swap/interpolate/scale/neutralize) so the interface claim is demonstrated by *operations*, not architecture prose (P0.3). (b) Optional multi-rate prosody (prosody at 6.25 Hz under 12.5 Hz content, `prosody_rate_divisor`) — prosody moves slower than phonetics; no factorized codec exploits rate asymmetry between streams. Ablation-gated, default off (P1.1) |
| CTC-on-quantized-content (upsample 2) | SiTok (ICLR'26) — same mechanism, scaled 1.6B/2M h | Derivative alone. Novel only as one side of **supervision asymmetry**: CTC pulls text INTO content while GRL pushes text OUT of prosody. Kanade/DisCodec use bottlenecks/hybrid losses, FACodec uses supervision+GRL but non-causal and with speaker, not prosody, as the pivot. | Name the mechanism once, ablate both sides (ctc.weight=0 arm exists; add leakage-weights=0 arm to the ablation grid — config-only). Never present CTC itself as ours |
| Prosody supervision (quantized F0 + energy) | FACodec (F0/energy supervision on prosody substream) | Not novel; it is the *citable precedent* the verdict protocol wants | none — cite FACodec as validation of the mechanism |
| Speaker as global reference vector (in neither stream) | DisCodec/FACodec have timbre *tokens* | Deliberate difference, must be argued: cloning requires explicit reference (responsible release) + keeps both streams speaker-free (leakage matrix proves it) | none — add the argument to the paper outline; leakage matrix speaker rows are the evidence |
| Block-causal shortcut flow-matching decoder, 1–2 NFE | SiTok (diffusion, multi-step, non-causal), DisCodec (GAN), TaDiCodec (diffusion, non-causal) | **Novel combination**: no factorized or semi-discrete tokenizer has a causal few-step decoder. This is also the "results better than baseline" lever for latency | Vocoder ceiling measurement FIRST (R5) — the fidelity tables die without it (P0.1) |
| Modelability protocol | STAB/DASB (task probes, no controlled generation probe); DC-Spin (n-gram perplexity); 2601.06329 (proves naive perplexity invalid) | **Novel if and only if** framed as the controlled answer to 2601.06329: fixed probe compute, bits-per-SECOND, probe→downstream transfer validation inside the same suite | Probe reports its own param/step budget so runs are provably matched (P0.4); protocol doc gains a "response to the fallacy critique" section |
| Matched internal baselines + RVQ/non-causal arms | none run this controlled grid | The fairness grid is itself reviewable substance | none — built |

**Net novelty statement (use this wording discipline everywhere):** the
contribution is the *conjunction* — causal, semi-discrete, factorized at
the interface — plus the controlled evaluation protocol. Any sentence
claiming novelty for a single ingredient will be matched to SiTok, FACodec,
VoxCPM, or Mimi by reviewers.

## 2. The win-set: where "better than baseline" is structurally achievable

We commit to beating baselines where the architecture gives us an edge,
and we pre-declare comparisons we will *not* win (credibility > bravado):

**Must-win (structural advantage, measured vs public checkpoints):**
1. **Fidelity at ≤300 bps discrete** — our 279 bps (22.3 bits @ 12.5 Hz)
   vs TaDiCodec (~88 bps, weaker recon) and Mimi@2-3 books (~275-410 bps).
   Nobody else has fidelity-competitive sub-300 bps *causal* tokens.
2. **Prosody correlates (F0 RMSE, voicing F1, energy corr, emotion-SIM)
   at matched bitrate** — we supervise F0; Mimi/X-codec2/WavTokenizer do
   not. Losing here means R1 (factorization collapse) fired — it's the
   same signal.
3. **Modelability bits/s per kbps** + downstream small-TTS at fixed LM
   compute vs Mimi tokens — single large-codebook stream + slow prosody
   stream vs 8 RVQ streams; DC-Spin's perplexity-vs-SLM correlation is
   the citable prior that this predicts downstream ability.
4. **First-audio latency & chunk-boundary continuity** vs every diffusion/
   GAN-offline decoder (SiTok, TaDiCodec, X-codec2) — only Mimi competes,
   and Mimi has no continuous view or prosody stream to trade against.
5. **Zero-shot language tokenization** (byte-CTC, DE/FR/JA/KO never seen)
   — no factorized-codec paper reports this at all.
6. **Prosody control** (swap/scale/neutralize with content WER preserved)
   — only DisCodec can attempt it, post-fusion and offline; direct
   head-to-head if their checkpoint releases, else vs FACodec.

**Will-not-win (say so in the paper, with the reason):**
- PESQ vs DAC@8 kbps or any ≥1 kbps waveform codec (28× our bitrate).
- WER-recovery vs 50 Hz semantic tokenizers with 4× our frame rate
  (X-codec2) on *reconstruction* ASR — we compete per-bit, not per-frame.
- Scaling-law claims vs SiTok's 1.6B/2M h — out of compute class; we
  compare mechanism, not scale, and say so.

## 3. Scope of change

**P0 — before any training job (cheap, protects everything):**
1. `scripts/vocoder_ceiling.py` — ground-truth mel → vocoder → PESQ/STOI
   upper bound on the eval set (R5). One GPU-hour; can invalidate table
   designs. BLOCKER for stage A.
2. Claim-wording updates in BASELINES_AND_EVAL (FACodec added to Tier 2 —
   its checkpoint is public and it is the *right* disentanglement
   baseline for the leakage matrix; DisCodec/ProsoCodec/Kanade rows in
   Tier 4 unless checkpoints appear).
3. `fact/edit.py` — prosody-stream operations (swap / interpolate /
   scale / neutralize) via code-space editing + lattice re-rounding.
   This is the RQ2 demo made executable, and it makes the semi-discrete
   duality *do* something: edits happen in the continuous view, land
   exactly on valid discrete states.
4. Modelability probe reports its parameter/step budget in results
   (matched-compute is provable, answering 2601.06329).
5. `scripts/tripwires.py` — one command computing the RISK §2 gate
   metrics from a checkpoint: per-stream codebook utilization/perplexity
   (R1), channel-dominance via channel-zeroed reconstruction deltas (R2),
   content→F0 vs prosody→F0 quick probes (R1). Run at every pilot
   checkpoint; the week-6 gate reads its JSON.

**P1 — pilot-scale decisions (config/architecture, ablation-gated):**
1. **Multi-rate prosody** (`prosody_rate_divisor: 2` → prosody at
   6.25 Hz): novelty-raiser (rate-asymmetric factorization) and bitrate
   win (−4.65 bits/frame ≈ −58 bps); risk: coarser F0 tracking — the
   F0-RMSE column decides. Default 1 (off) for the main arm; one pilot
   ablation decides promotion.
2. λ_ctc 3-point sweep at 5K h before committing the big run (R3).
3. Leakage-weights=0 arm added to the ablation grid (config-only) to
   isolate GRL's contribution from CTC asymmetry.

**P2 — deferred, do not build now:**
- Waveform-native decoder head (only if vocoder ceiling fails R5 after
  stage-C fine-tune).
- Forced-aligner duration/voice-quality prosody targets (plan already
  defers; F0+energy has FACodec precedent).
- FACodec numerical wrapper (stub now; full wrapper when its runtime
  deps are validated on Leonardo — do not burn sandbox time on an
  untestable integration).

## 4. What does NOT change

Encoder/decoder scaffolding, FSQ lattices and bitrates, the matched
baseline grid, CINECA workflow, the padding/masking semantics, and the
verdict protocol. The literature check sharpened claims; it did not
invalidate the architecture. No mechanism in the pipeline was shown
redundant — each now has either a novelty role or a citable-precedent
role, which is exactly the position to train from.
