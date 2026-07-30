# FaCT — Factorized Causal Tokenizer

A **semi-discrete, causal, content/prosody-factorized** speech representation
for TTS. One representation with four properties no existing tokenizer
combines:

1. **Acoustically near-lossless** — a continuous view with no hard
   quantization ceiling (FSQ pre-rounding vectors + optional continuous
   residual channel).
2. **Discretely addressable** — every frame also has exact integer indices
   *of the same state* (the FSQ lattice point), so LM prediction, editing,
   and RL log-probs all work.
3. **Causal / streamable** — strictly causal encoder, block-causal
   shortcut flow-matching decoder distillable to 1–2 NFE.
4. **Explicitly factorized** — separate content and prosody token streams,
   with speaker identity in *neither* (global reference conditioning only).

The three mechanisms being combined: VoxCPM's FSQ-as-regularization
(semi-discrete duality), SiTok's CTC-on-quantized-tokens semantic grounding
(no SSL teacher at inference), and DSA-style factorization extended to
prosody via asymmetric supervision + gradient-reversal leakage penalties.

## Architecture

```
24 kHz wav → log-mel 50 Hz → causal encoder (~119M) → 12.5 Hz frames
    → factorized semi-discrete bottleneck
        CONTENT head: FSQ 8·8·8·4·4 = 8192 codes (2^13) + causal CTC head
        PROSODY head: FSQ 5^4 = 625 codes (~2^9.3), F0/energy supervised
        RESIDUAL:     16-dim continuous, KL-regularized (optional)
    → block-causal shortcut flow-matching decoder (~230M) → mel
    → vocoder (reuse open causal Vocos/HiFi-GAN; out of scope here)
```

## Repository map (↔ research plan)

| Plan item | Code |
|---|---|
| FSQ dual view (§3.1) | `fact/modules/fsq.py` — `continuous` / `quantized` / `indices` of one state |
| CTC-on-content grounding (§3.2) | `fact/modules/heads.py::CTCHead` |
| Factorization pressure (§3.3) | asymmetric losses in `fact/model.py::training_step` + `heads.py::LeakageProbes` (GRL) |
| Speaker outside both streams (§3.3) | `fact/modules/speaker.py` (global reference vector) |
| Continuous residual (§3.4) | `fact/modules/bottleneck.py` (`residual_dim`, KL) |
| Causal 1–2-step decoder (§3.5) | `fact/modules/decoder.py` — block-causal attn, `shortcut_loss`, NFE-matched sampling |
| 12.5 / 6.25 Hz variants (§3.6, RQ4) | `configs/fact_base.yaml` / `configs/fact_6hz.yaml`; token-count-fixed crops via `--crop-tokens` |
| Stage A / B training (§5) | `fact/train/stage_a.py`, `fact/train/stage_b_shortcut.py` |
| Preprocessing (mel/F0/text) | `fact/data/preprocess.py` (pyworld; byte-level text) |
| Modelability protocol (§6.1) | `fact/eval/modelability.py` — LM-probe bits/frame + bits/second |
| Leakage matrix (§6.3) | `fact/eval/leakage.py` — post-hoc probes on frozen features |
| Streaming fitness (§6.4) | `fact/eval/reconstruction.py::chunk_boundary_discontinuity` |
| Fidelity ceiling (§6.5) | `fact/eval/reconstruction.py` — mel dist, F0 RMSE, voicing F1, energy corr + hooks for UTMOS/PESQ/WER/SIM |

## Quick start

```bash
pip install -e ".[dev]"          # core: torch, numpy, pyyaml
pytest                            # 29 CPU tests: shapes, causality, FSQ round-trip, training smoke

# CPU smoke run (synthetic speech-like data, tiny model):
python -m fact.train.stage_a --tiny --synthetic --steps 50

# Real run:
python -m fact.data.preprocess --manifest train.tsv --out shards/ --config configs/fact_base.yaml
python -m fact.train.stage_a --config configs/fact_base.yaml --shards shards/ --ckpt-dir runs/a
python -m fact.train.stage_b_shortcut --config configs/fact_base.yaml \
    --init runs/a/step_0400000.pt --shards shards/ --ckpt-dir runs/b
```

## Interfaces

```python
from fact import FaCT, FaCTConfig

model = FaCT(FaCTConfig())
toks = model.tokenize(wav=wav)            # or mel=mel
toks.content_indices                      # (B, T) ints  — discrete view
toks.content_continuous                   # (B, T, 5)    — continuous view, same state
toks.prosody_indices, toks.residual       # prosody stream + continuous residual

mel = model.detokenize(toks.content_indices, toks.prosody_indices,
                       ref_mel=ref, residual=toks.residual, nfe=2)

# RQ2 demo — zero-cost prosody transfer: content from A, prosody from B:
mel = model.detokenize(toksA.content_indices, toksB.prosody_indices, ref_mel=refA)
```

## Verified properties (tests)

- `tests/test_causality.py` — perturbing future audio never changes past
  tokens (encoder) or past blocks (decoder). Causality is checked
  mechanically, not asserted.
- `tests/test_fsq.py` — index ↔ lattice round-trip is exact; all codes
  reachable; straight-through gradients flow; the continuous and quantized
  views are within half a lattice cell.
- `tests/test_model.py` / `test_decoder.py` — full stage-A loss is finite
  and backprops; stage B freezes the tokenizer; NFE ∈ {1,2,4} sampling;
  prosody-swap interface.
- `tests/test_eval.py` — the modelability probe recovers ~log2(V) bits on
  random streams and ~0 bits on constant streams; leakage probes separate
  informative from uninformative features.

## Status / roadmap

This is the week-1–6 scaffold of the research plan (pipeline validated
end-to-end at tiny scale on synthetic data). Next: 5K-h Emilia slice run,
codebook-utilization + CTC-convergence checks, small-scale RQ3 probe
(FaCT vs Mimi vs matched continuous VAE), then the full stage-A run.
Matched internal baselines (single-FSQ, pure-VAE) are config-level
ablations: set `prosody_levels`/`residual_dim` and loss weights accordingly.
