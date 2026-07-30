# Baselines & Evaluation Strategy (ICLR submission)

Emilia is multilingual (EN/ZH/DE/FR/JA/KO), so the baseline set fixes the
scope: we train where the strongest baselines are comparable, and use the
remaining languages as a free zero-shot generalization eval.

## 0. Scope decision

- **Training languages: EN + ZH** (~47K h + ~50K h in Emilia; every serious
  external baseline covers at least one of these, most cover both).
  Byte-level CTC makes the content stream language-agnostic by construction
  — no per-language text frontend.
- **Held-out languages: DE / FR / JA / KO** — *never trained on*, used only
  to test zero-shot tokenization (reconstruction + probe). Costs zero extra
  training and gives ICLR a multilingual-generalization claim nobody's
  tokenizer paper currently makes cleanly.
- **Expressive/emotion eval:** Expresso + ESD (EN/ZH emotional speech) —
  where the prosody stream must show its value (RQ2).

## 1. Baseline taxonomy

ICLR reviewers weigh **controlled** comparisons above leaderboard rows.
So: Tier 1 (retrained, identical everything) carries the claims; Tier 2
(public checkpoints) provides context; Tier 3 anchors the downstream table;
Tier 4 is positioning only (related work, no numeric comparison claimed).

### Tier 1 — Matched internal baselines (retrained; identical encoder/decoder/data/steps)

These isolate one design decision each. All share our encoder, decoder,
CTC head, data, and compute; only the named component changes.

| Arm | What changes | Claim it isolates | Status |
|---|---|---|---|
| **FaCT** (full) | — | — | implemented (`configs/fact_base.yaml`) |
| **Single-FSQ** | one FSQ over the *union* lattice (8·8·8·4·4·5·5·5·5): exactly matched bits (2^22.3) and dims (9), no prosody stream, no factorization losses | factorization itself (RQ2) | implemented (`configs/baseline_single_fsq.yaml`) |
| **Matched VAE** | pure-continuous KL latent, 25 dims (= 5+4+16), no discrete view | discreteness itself; the continuous side of RQ1/RQ3 | implemented (`configs/baseline_vae.yaml`) |
| **RVQ-swap** | replace the content FSQ with RVQ at matched bits (2 × 2^11 or 4 × 2^6 sweep) | extends 2509.20060's FSQ≻RVQ finding to the tokenizer level (ablation §7) | to implement (small: an RVQ module behind `variant`) |
| **No-CTC** | `ctc.weight: 0` | semantic grounding's contribution to modelability | config-only |
| **No-residual** | `residual_dim: 0` | the discrete-vs-continuous figure; also answers the "hidden continuous side-channel" attack (see §3) | config-only |
| **Non-causal** | bidirectional encoder + full-attention decoder | the price of streamability | to implement (mask flag) |

### Tier 2 — External public tokenizers (no retrain; reconstruction + modelability probe)

Chosen for *relevance to a specific claim*, not popularity. Each row states
why it earns its place.

| Baseline | Frame rate / discrete bitrate | Causal? | Semantic grounding | Langs | Why it's in |
|---|---|---|---|---|---|
| **Mimi** (Kyutai) | 12.5 Hz, ~1.1 kbps (8 RVQ books) | yes (enc+dec) | WavLM-distilled 1st book | mostly EN/FR | *The* reference point: same frame rate, causal, split semantic/acoustic — everything we do, minus duality and factorization. Wrapper implemented. |
| **X-codec2** | 50 Hz, ~800 bps (1 × 65k book) | no | semantic+acoustic fusion | EN/ZH | The single-codebook semantic-fusion school (LLaSA line); the strongest "one merged stream" counterpoint to factorization. |
| **WavTokenizer** | 40–75 Hz, ~480–900 bps (1 × 4k) | no | none | EN | The extreme-compression single-VQ school; popular downstream target. |
| **TaDiCodec** | 6.25 Hz, ~88 bps | no (diffusion dec) | text-aware training | EN/ZH | Closest decoder philosophy (diffusion/flow decode, text conditioning); also the 6.25 Hz point for RQ4 context. |
| **DAC** (Descript) | 86 Hz, 8 kbps (RVQ-9) | no | none | any (acoustic) | Pure-fidelity ceiling reference: what "no semantic constraint, lots of bits" buys. Anchors the reconstruction axis. |
| **SpeechTokenizer** | 50 Hz, RVQ-8, HuBERT-distilled | no | SSL teacher | EN | The classic semantic-distillation recipe — the "needs an SSL teacher" contrast to our CTC-on-tokens. Optional if space is tight. |
| **NanoCodec** (NVIDIA) | 12.5 Hz-class, FSQ, causal | yes | none | EN-centric | The closest *causal FSQ* codec — directly adjacent design without duality/factorization. Include if the NeMo checkpoint is usable. |

Coverage caveat, stated in the paper: EN-only baselines are evaluated on EN
(and reported as "—" on ZH) rather than punished out-of-domain. Main
cross-tokenizer tables are EN; ZH tables include only ZH-capable baselines.

### Tier 3 — Downstream TTS anchors (RQ3 centerpiece table)

Same 0.3B LM recipe, same 10K h, same token budget, per stream:

1. FaCT — **indices** (AR over content ⊕ prosody, chain-rule within frame)
2. FaCT — **continuous** (LM regresses the continuous view via a small flow/MSE head)
3. FaCT — **hybrid** (AR content indices + small head for prosody/residual)
4. Matched VAE — continuous (the strongest continuous-latent arm, same stack)
5. **Mimi** tokens (external discrete anchor)
6. X-codec2 tokens (second external anchor; drop first if compute-bound)
7. Single-FSQ — indices (does factorization help *downstream*, not just probes)

Rows 1–4 and 7 are fully controlled; rows 5–6 anchor against the field.

### Tier 4 — Positioning only (no numeric head-to-head claimed)

VoxCPM (semi-discrete as internal trick), dots.tts AudioVAE / HoliTok
(continuous line), Qwen3-TTS dual tokenizers, DSA-Tokenizer
(content/speaker factorization), SiTok (CTC mechanism, scaling). These are
either full TTS systems without a clean tokenizer interface, or not open —
compare in related work / discussion, not in tables. Overclaiming
comparability here is a rebuttal wound; framing them as motivation is a
strength.

## 2. Evaluation strategy — five suites mapped to RQs

### Suite A: Reconstruction fidelity (RQ1)
- **Sets:** LibriTTS-R test-clean; Emilia held-out EN/ZH; **zero-shot** Emilia DE/FR/JA/KO; Expresso; ESD.
- **Metrics:** PESQ-WB, STOI, UTMOSv2 (naturalness), WER (Whisper-large-v3, multilingual — one ASR for all languages), speaker-SIM (WavLM-base+ ECAPA), emotion-SIM (emotion2vec), F0 RMSE + corr, voicing F1, energy corr. The prosody correlates and emotion-SIM are the axes standard metrics miss — that's where factorized codecs usually silently lose.
- **Fairness:** report per-tokenizer *discrete bitrate* alongside every number; group comparisons at ≤300 bps / ≤1.1 kbps tiers. FaCT rows: **discrete-only** (279 bps: 13 + 9.3 bits @ 12.5 Hz) *and* **+residual** — the residual is continuous side information and must be disclosed as such, with the no-residual ablation closing the loophole.
- **Speaker-reference disclosure:** our decoder takes a reference; reconstruction uses the utterance itself as reference (standard for factorized codecs), plus a cross-utterance-same-speaker condition to show it isn't leaking content through the reference.

### Suite B: Modelability (the protocol we release)
- LM-probe bits-per-frame **and bits-per-second** (frame-rate-normalized), fixed 0.1B probe, fixed data/token budget, 3 seeds, mean±std. Held-out NLL on speaker-disjoint shards.
- Run on: FaCT streams, single-FSQ, Mimi (8 books), X-codec2, WavTokenizer, TaDiCodec. The VAE arm has no bits — reported in Suite D instead (that *is* the discrete-vs-continuous story).

### Suite C: Disentanglement (RQ2)
- **Leakage matrix** (post-hoc probes on frozen features): content→F0, prosody→text, content→speaker, prosody→speaker; against the single-FSQ arm as the "everything leaks" reference.
- **Prosody transfer:** swap prosody stream between utterance pairs; measure F0-contour correlation with the *donor* vs text-WER preservation from the *content source*; emotion transfer accuracy on ESD (emotion2vec classifier).
- This suite only exists because of factorization — it's the differentiating demo.

### Suite D: Downstream TTS per unit LM compute (RQ3 — headline table)
- Tier-3 arms; eval on Seed-TTS-eval (EN+ZH; WER + SIM) and an EmergentTTS-Eval subset (expressive); identical LM FLOPs per arm; 2–3 seeds on the small scale, mini scaling check (0.1B→0.3B) to show trend stability.

### Suite E: Streaming fitness
- First-audio latency (encoder lookahead 0 + decoder block × NFE), RTF, chunk-boundary discontinuity (ΔF0/Δenergy at block edges vs within — implemented), quality-vs-chunk-size curve, causal vs non-causal arm gap, NFE ∈ {1, 2, 4, 32}.

## 3. Pre-empting the reviews (ICLR-specific)

| Expected attack | Our answer (built into the design) |
|---|---|
| "Public checkpoints saw different data — unfair" | Claims rest on Tier 1 (identical data/compute); Tier 2 is context, labeled as such. |
| "The continuous residual is a hidden side-channel" | Disclosed bitrate accounting + no-residual arm in every table. |
| "Speaker reference leaks information" | Cross-utterance reference condition + speaker row of the leakage matrix. |
| "Cherry-picked bitrate comparisons" | Bitrate column everywhere; tiered groupings; no cross-tier boasting. |
| "Factorization is just multitask supervision" | Single-FSQ arm gets the same F0/energy losses on its unified stream as an extra ablation — separates *supervision* from *separation*. |
| "MOS is noisy" | UTMOSv2 primary + small CMOS study only for the final system pair; everything else objective. |
| "Probe results won't transfer to real TTS" | Suite B (probe) and Suite D (actual small TTS) reported side by side — correlation between them is itself a finding of the protocol. |

## 4. Priority order under compute pressure

1. Tier 1: FaCT, single-FSQ, VAE (the paper stands on these three) + config-only arms.
2. Suite B on FaCT/single-FSQ/Mimi + Suite D rows 1–4.
3. Tier 2 externals beyond Mimi (checkpoints, cheap — only eval compute).
4. RVQ-swap and non-causal arms.
5. Zero-shot language suite (eval-only, cheap — keep late but keep).
6. NanoCodec/SpeechTokenizer rows, EmergentTTS subset, CMOS study.

Cut from the bottom; never cut a Tier-1 arm — a missing controlled baseline
is the one reviewers won't forgive.
