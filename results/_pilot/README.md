# Pilot and smoke results (outside the frozen result set)

These are real runs, but **not** paper results:
- they come from earlier code versions, shorter training (25–100K steps) or a
  pilot dev split (~28.8K dev tokens, versus 28.6K for the final dev shards);
- or they use a 200-utterance LibriSpeech subset instead of the full 2,620.

They are kept for the record and must not be quoted as headline numbers. Any
of them used in the paper (e.g. the residual-capacity or λ_CTC pilot sweeps)
must be labelled as a pilot study with its setting.

| Path | What |
|---|---|
| `baseline_recon_ls200/` | baseline smoke evals on 200 LibriSpeech test-clean utterances |
| `vocoder_ceiling_ls200.json` | vocoder copy-synthesis ceiling on 200 utterances (the full one is `results/vocoder_ceiling_full.json`) |
| `fact_recon/pilot_*` | pilot-run reconstructions on 200 utterances |
| `tripwires/pilot_*`, `tripwires/step_*` | tripwires of the pilot run (pilot dev split) |
| `tripwires/residual_*` | residual-capacity / residual-dim pilot sweep (50K steps, pilot dev split) |
| `tripwires/main_00*`, `tripwires/v2_nodrop_*`, `tripwires/v3_nodrop_*` | early fact_main checkpoints (25–75K) re-evaluated under older tripwire code; superseded by `results/tripwires/fact_main_*` |
| `lambda_ctc_sweep/` | λ_CTC pilot sweep (50K steps) |

Moved here on 2026-11-03 by the pre-freeze audit (docs/AUDIT_2026-11-03.md, item A7).
