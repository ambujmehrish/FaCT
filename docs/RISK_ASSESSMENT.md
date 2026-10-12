# Risk Assessment — read before submitting any training job

*Last literature check: 2026-09-30. Re-run the §1 searches before locking
paper claims; this field moves monthly.*

## 0. Ground rules this document operationalizes

1. **SOTA cannot be guaranteed; it can be engineered for.** What we control
   is (a) which claims we stake (first-mover claims beat leaderboard-delta
   claims), (b) how early we detect a losing arm and redirect compute, and
   (c) comparison scope discipline. Every risk below therefore carries a
   *tripwire* (earliest observable signal), a *pivot* (what to do with the
   remaining compute), and a *SOTA-preserving fallback claim*. The goal is
   that no plausible outcome leaves us holding only a negative result —
   because we rerouted at 50–100K steps, not because we buried anything:
   whatever we ultimately publish reports what actually happened.
2. **No verdict without citable grounding** (the verdict protocol, §4):
   before declaring any result — good or bad — we find published work that
   (a) used the same analysis and (b) establishes its validity and known
   failure modes. A verdict that can't cite its method is not a verdict.

## 1. Competitive landscape — what today's check found (claim-critical)

| Plan claim | Status after search | Required repositioning |
|---|---|---|
| "Nobody has a prosody stream" (RQ2) | **DEAD as stated.** [DisCo-Speech/DisCodec](https://arxiv.org/abs/2512.13251) (Dec 2025) factorizes content/prosody/timbre; [ProsoCodec](https://arxiv.org/pdf/2606.21888) (Jun 2026) is a prosody-oriented codec; [FACodec/NaturalSpeech 3](https://arxiv.org/pdf/2601.09239) (2024, cited by DSA) already factorized prosody with GRL; [Kanade](https://arxiv.org/pdf/2602.00594) (Feb 2026) does single-stream disentanglement. | Our claim becomes: **first *causal/streamable, semi-discrete* tokenizer that exposes prosody as a *separately addressable stream at the LM interface*.** Crucially, DisCodec *fuses* content+prosody back into unified tokens before LM prediction — the separate-stream-at-the-interface property (swap/edit prosody without touching content tokens, hybrid RQ3 interfaces) survives, and FACodec/DisCodec are non-causal. DisCodec, ProsoCodec, Kanade, FACodec must all be cited, compared where checkpoints exist, and the intro must not contain the word "first" without these qualifiers. |
| "The field has no modelability metric" (§6) | **Weakened.** [STAB](https://arxiv.org/pdf/2409.02384) and [DASB](https://arxiv.org/pdf/2406.14294) benchmark speech tokenizers; [DC-Spin](https://arxiv.org/pdf/2410.24177) already reports n-gram perplexity over token streams and correlates it with SLM tasks; [TokEval](https://arxiv.org/pdf/2608.18062) (Aug 2026) is a *text*-tokenizer suite (not speech — our lane is clear there) whose intrinsic-metrics-predict-LM-ability result (ρ≈0.8) is a design template. **Critically:** [On the Fallacy of Global Token Perplexity](https://arxiv.org/html/2601.06329v1) (Jan 2026) argues naive cross-tokenizer perplexity comparison is invalid for speech. | Our protocol is still unclaimed — a *controlled fixed-compute LM-probe* with bits-per-**second** normalization and a probe→downstream-TTS transfer correlation — but it must be built **as an answer to 2601.06329's critique** (cite it, show our normalization addresses each failure mode it names) and positioned against STAB/DASB (task-probe benchmarks, no controlled generation-side probe). If we ignore that paper, reviewers will use it against us; if we engage it, it is our motivation section. |
| SiTok's CTC mechanism | Confirmed as expected: [SiTok, ICLR 2026](https://arxiv.org/abs/2602.06602) — CTC on quantized latents, 12.5 Hz, 0.2 kbps, 1.6B/2M h. | Our novelty is CTC-as-*asymmetric factorization pressure*, never CTC itself. Any sentence implying we invented text-grounded quantization gets rejected by SiTok's own reviewers. |
| FSQ > RVQ premise | Supported: [NeuCodec](https://www.emergentmind.com/topics/neucodec), [FSQ-optimality](https://arxiv.org/abs/2606.09962). Our RVQ arm tests it at matched bits anyway. | None. |
| 6.25 Hz cliff = training-length artifact (RQ4) | Confirmed live and *crowding*: [2606.16969](https://arxiv.org/pdf/2606.16969) (the claim itself), [FlexiSLM](https://arxiv.org/pdf/2606.31247) (dynamic frame rates), [frame-rate case study](https://arxiv.org/html/2505.17076v3). | RQ4 stays an *embedded* replication (expressive/multilingual angle is still open); it is no longer safe as a standalone Paper 2 — assume someone publishes the plain replication within months. |
| Semi-discrete duality (RQ1) | [VoxCPM](https://arxiv.org/abs/2509.24650) line advancing: [VoxCPM2](https://arxiv.org/html/2606.06928v1) exists. Duality still internal there (tokenizer-free TTS), not an exposed tokenizer interface. | **Read VoxCPM2 in full before locking RQ1 wording.** Highest scoop risk of the four properties; the four-properties-in-one-tokenizer conjunction is the defensible claim, not any single property. |

**Standing order:** weekly arXiv sweep on {speech tokenizer, factorized
codec, semi-discrete, prosody tokens, frame rate} — 30 minutes that can
save a rejection cycle. Any new hit gets a row in this table.

## 2. Risk register — technical (each with tripwire → pivot → fallback claim)

**R1. Factorization collapse** (prosody stream ignored, or content leaks
prosody). *Severity: high. Likelihood: medium.*
- Tripwire (≤50K steps, 5K-h pilot): prosody codebook perplexity < ~50 of
  625; leakage probe content→F0 within 5 points of the single-FSQ arm;
  prosody-swap does not move F0 contour correlation.
- Pivot: raise GRL λ / add InfoNCE penalty (FACodec's recipe is citable
  precedent for both); if still collapsed at second gate, drop to
  single-FSQ as the *main* model.
- Fallback claim: causal semi-discrete tokenizer + modelability protocol
  (RQ1+RQ3 stand alone; DisCodec itself shows partial-disentanglement
  papers publish fine when the rest is strong).

**R2. Residual channel swallows everything** (decoder reads the 16-dim
continuous channel, ignores discrete streams — the inverse of posterior
collapse; our KL is per-frame on a *deterministic-ish* channel, so
information can hide there).
*Severity: high — it silently invalidates every "discrete interface"
claim while reconstruction looks great. Likelihood: medium.*
- Tripwire: at every eval checkpoint, decode with residual zeroed; if
  mel-L1 degradation from zeroing residual > 3× the degradation from
  zeroing prosody, the residual is dominant. Also train the leakage probe
  residual→text.
- Pivot: raise `residual_kl_weight` (β-anneal; cite β-VAE precedent),
  shrink `residual_dim` 16→8, or residual-dropout during decoder training.
- Fallback claim: report the residual-capacity sweep as the
  discrete-vs-continuous figure — this ablation is *already in the plan*;
  the risk only materializes as a finding if undetected.

**R3. CTC starves acoustics** (λ_ctc too high → encoder becomes an ASR
front-end, reconstruction plateaus; too low → content stream not
grounded). *Severity: medium. Likelihood: high — SiTok says λ is crucial
and does not publish the value.*
- Tripwire: recon loss plateau while CTC keeps dropping (or vice versa)
  in the first 50K steps; greedy-CTC CER > 40% at 100K steps means no
  grounding.
- Pivot: the λ_ctc sweep is ablation #2 — run it at pilot scale *first*
  (3 points, 5K h) instead of trusting λ=1.0 for the big run.
- Fallback: the sweep itself is a service-to-the-field contribution.

**R4. Shortcut-distilled 1–2 NFE causal decoder quality gap.**
*Severity: medium (latency claim, not representation claims).*
- Tripwire: stage-B UTMOS drop > 0.3 vs 32-NFE at pilot scale.
- Pivot: ship 4-NFE (still <80 ms budget at block 160 ms? — recompute
  honestly) and/or the NanoCodec-style GAN head.
- Fallback: all four representation properties are NFE-independent.

**R5. Mel + vocoder loses to end-to-end waveform codecs on PESQ/UTMOS**
(we decode to mel and vocode; Mimi/DAC emit waveforms — their metrics
include no vocoder floor, ours do).
*Severity: high for RQ1 tables. Likelihood: medium-high.*
- Tripwire: *before any training*, measure the vocoder ceiling: run
  ground-truth mel through the chosen vocoder (Vocos) on the eval set;
  that PESQ/UTMOS is our upper bound. If the ceiling is below Mimi's
  reconstruction numbers, the comparison is lost before we start.
- Pivot: fine-tune the vocoder (stage C exists in the plan) or move the
  decoder to a waveform head; report "codec-only" (mel-domain) and
  "system" (with vocoder) rows separately — citable precedent: every
  mel-domain TTS paper separates acoustic-model and vocoder error.
- This measurement costs one GPU-hour. **Do it first.**

**R6. Emilia transcript noise breaks CTC grounding** (Emilia transcripts
are ASR-generated; byte-level CTC on noisy transcripts = noisy gradient;
`ctc_infeasible` guards length, not correctness).
*Severity: medium. Likelihood: medium.*
- Tripwire: pilot-scale greedy-CTC CER vs LibriTTS-R (clean) minus Emilia
  (auto): a gap ≫ the corpora's known ASR-quality gap means transcript
  noise is the binding constraint.
- Pivot: weight CTC by corpus, or filter Emilia by its released
  DNSMOS/confidence fields.

**R7. Byte-level CTC on Chinese** (UTF-8 gives 3 bytes/char; the CTC must
learn multi-byte composition; upsample=2 gives 25 pos/s vs ~12-15 bytes/s
for Mandarin — feasible, but composition is the risk, not length).
*Severity: medium (ZH tables only).*
- Tripwire: ZH greedy-CTC CER at pilot scale ≫ EN.
- Pivot: char-level vocab for ZH (vocab ~6-8K) — shard fingerprint already
  guards the re-preprocess this requires.

**R8. RVQ arm artifacts** (EMA under DDP, dead codes at 91-way books) —
mitigated in code (all-reduced stats, reseeding), but *watch* `commit`
loss stability; RVQ instability is itself the citable expected result
(NeuCodec), so this arm cannot produce an uninterpretable outcome.

**R9. Infrastructure** — Leonardo queue latency on 2-node jobs, 24 h wall
cap (mitigated: auto-resume chaining), $WORK quota (60K h of fp16 mel
shards ≈ 4-5 TB — *check quota before preprocessing*, or store mel as
int8-quantized + dequant in the loader if tight), single-node fallback if
2-node queue times dominate (grad_accum 2 restores the batch).

## 3. The two structurally dangerous outcomes (and why they're covered)

1. **"Continuous VAE wins everything" (RQ3).** Then the paper is the
   controlled evidence *settling a live debate* — that is a headline
   result (the field currently argues both sides from uncontrolled
   comparisons), not a negative result, and the semi-discrete model's
   continuous view ties the VAE arm by construction (same training
   objective family) — the discrete view is additive capability, so FaCT
   is never *behind* its own VAE arm on the continuous interface.
2. **"Factorization doesn't help downstream" (RQ2/RQ3).** Then the
   prosody-swap *capability* results (control, editing) carry RQ2 — a
   capability nobody disputes is useful — and the per-FLOP table carries
   RQ3 regardless of which interface wins. The claim structure was chosen
   so the centerpiece table is informative under every ordering.

The one outcome with no strong paper: **all four properties
simultaneously mediocre** — fidelity below Mimi at matched bitrate AND no
factorization AND no modelability edge. The week-6 gate exists precisely
to make us stop and rethink at pilot cost, not at 10-GPU-week cost.
If the pilot hits that gate, we do not scale up and hope.

## 4. Verdict protocol (mandatory before any claim, internal or in-paper)

For every result we intend to state as a finding:
1. **Method precedent:** name ≥1 published work using the same analysis
   (metric, probe design, statistical test) for the same kind of claim;
   if none exists, the *method* needs its own validation experiment first.
   Known constraints on our methods, found today: cross-tokenizer token
   perplexity is criticized by [2601.06329](https://arxiv.org/html/2601.06329v1)
   — our bits/second probe must explicitly answer it; GRL-based
   disentanglement probes follow FACodec/DSA precedent; token-count-fixed
   training follows [2606.16969](https://arxiv.org/pdf/2606.16969).
2. **Baseline sanity:** our measured numbers for public checkpoints must
   fall in published ranges (docs/CINECA.md Step 1) *before* any table
   containing them is trusted.
3. **Magnitude check:** compare effect size against the closest published
   effect (e.g., a 0.1 UTMOS delta is within reported seed noise for
   small TTS models — find the citation before calling it a win or loss).
4. **Seeds/CI:** probe-scale results: 3 seeds, mean±std; a claim that
   flips within the CI is not a claim.
5. **Adversarial pass:** for every headline number, write down the
   strongest alternative explanation (side-channel, data mismatch,
   bitrate mismatch) and the ablation that excludes it — §3 of
   BASELINES_AND_EVAL already lists the known ones.

## 5. Revised pre-training checklist (order matters)

1. Vocoder-ceiling measurement (R5) — 1 GPU-hour, can kill a table design.
2. Read VoxCPM2 + DisCo-Speech + ProsoCodec in full; update §1 and claim
   wording (0 GPU-hours, protects the whole paper).
3. Baseline eval job (Step 1) — validates environment + populates the
   sanity ranges the verdict protocol needs.
4. λ_ctc 3-point sweep at 5K-h pilot scale (R3) *before* the main run.
5. Pilot with full tripwire logging (R1, R2, R6, R7 metrics at 50K steps).
6. Week-6 gate review against §3 — then, and only then, the 8-GPU
   stage-A arms.
