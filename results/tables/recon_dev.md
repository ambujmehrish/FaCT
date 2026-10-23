## Emilia-EN dev (2,000 utterances, model-selection set)

Mean [95% bootstrap CI] over utterances (n = 2000). WER is corpus-level vs the ground-truth transcript. mel-L1 uses bins below the source audio's Nyquist. FaCT rows are pre-vocoder (mel domain): waveform metrics are n/a until the stage-C vocoder exists.

| system | kbps | PESQ-WB ↑ | STOI ↑ | WER % ↓ | mel-L1 ↓ | F0 RMSE ↓ | V/UV F1 ↑ | energy r ↑ | UTMOS ↑ | SIM ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| FaCT fact_main (tokens + residual) | 0.423 | n/a | n/a | n/a | 1.338 [1.331, 1.345] | n/a | n/a | 0.966 [0.965, 0.967] | n/a | n/a |
| FaCT fact_main (tokens only) | 0.279 | n/a | n/a | n/a | 1.848 [1.838, 1.858] | n/a | n/a | 0.880 [0.877, 0.882] | n/a | n/a |
| FaCT fact_main (tokens only (residual removed)) | 0.279 | n/a | n/a | n/a | 4.943 [4.914, 4.970] | n/a | n/a | 0.119 [0.116, 0.123] | n/a | n/a |
| FaCT main_c16_rd05 (tokens + residual) | 0.567 | n/a | n/a | n/a | 1.269 [1.262, 1.276] | n/a | n/a | 0.974 [0.973, 0.974] | n/a | n/a |
| FaCT main_c16_rd05 (tokens only) | 0.279 | n/a | n/a | n/a | 1.875 [1.865, 1.885] | n/a | n/a | 0.880 [0.878, 0.883] | n/a | n/a |
| FaCT main_c16_rd05 (tokens only (residual removed)) | 0.279 | n/a | n/a | n/a | 1.629 [1.619, 1.638] | n/a | n/a | 0.916 [0.915, 0.918] | n/a | n/a |
| FaCT main_c8_rd03 (tokens + residual) | 0.423 | n/a | n/a | n/a | 1.354 [1.347, 1.362] | n/a | n/a | 0.964 [0.963, 0.965] | n/a | n/a |
| FaCT main_c8_rd03 (tokens only) | 0.279 | n/a | n/a | n/a | 1.797 [1.786, 1.807] | n/a | n/a | 0.885 [0.882, 0.887] | n/a | n/a |
| FaCT main_c8_rd03 (tokens only (residual removed)) | 0.279 | n/a | n/a | n/a | 1.622 [1.613, 1.631] | n/a | n/a | 0.917 [0.915, 0.919] | n/a | n/a |
| FaCT main_c8_rd05 (tokens + residual) | 0.423 | n/a | n/a | n/a | 1.377 [1.369, 1.384] | n/a | n/a | 0.962 [0.961, 0.963] | n/a | n/a |
| FaCT main_c8_rd05 (tokens only) | 0.279 | n/a | n/a | n/a | 1.765 [1.755, 1.775] | n/a | n/a | 0.890 [0.887, 0.892] | n/a | n/a |
| FaCT main_c8_rd05 (tokens only (residual removed)) | 0.279 | n/a | n/a | n/a | 1.610 [1.601, 1.619] | n/a | n/a | 0.919 [0.917, 0.921] | n/a | n/a |
