## LibriSpeech test-clean (full, 2,620 utterances)

Mean [95% bootstrap CI] over utterances (n = 2620). WER is corpus-level vs the ground-truth transcript. mel-L1 uses bins below the source audio's Nyquist. FaCT rows are pre-vocoder (mel domain): waveform metrics are n/a until the stage-C vocoder exists.

| system | kbps | PESQ-WB ↑ | STOI ↑ | WER % ↓ | mel-L1 ↓ | F0 RMSE ↓ | V/UV F1 ↑ | energy r ↑ |
|---|---|---|---|---|---|---|---|---|
| mimi_n_streams2 | 0.275 | 1.25 [1.24, 1.25] | 0.767 [0.765, 0.768] | 8.06 [7.69, 8.46] | 1.846 [1.840, 1.852] | 32.5 [31.6, 33.4] | 0.677 [0.671, 0.682] | 0.953 [0.952, 0.953] |
| mimi_n_streams4 | 0.550 | 1.64 [1.64, 1.65] | 0.854 [0.852, 0.855] | 4.45 [4.19, 4.72] | 1.450 [1.446, 1.454] | 17.0 [16.4, 17.6] | 0.778 [0.774, 0.783] | 0.973 [0.973, 0.974] |
| dac_n_quantizers1 | 0.750 | 1.18 [1.18, 1.18] | 0.746 [0.744, 0.747] | 24.68 [23.85, 25.58] | 1.408 [1.403, 1.414] | 30.3 [29.4, 31.1] | 0.565 [0.559, 0.572] | 0.963 [0.962, 0.964] |
| xcodec2 | 0.800 | 2.31 [2.29, 2.32] | 0.913 [0.912, 0.914] | 3.30 [2.76, 3.99] | 1.063 [1.060, 1.067] | 14.7 [14.1, 15.3] | 0.815 [0.810, 0.819] | 0.993 [0.993, 0.993] |
| mimi_n_streams8 | 1.100 | 2.25 [2.24, 2.26] | 0.906 [0.904, 0.907] | 2.63 [2.44, 2.83] | 1.233 [1.230, 1.235] | 13.3 [12.8, 13.8] | 0.829 [0.825, 0.833] | 0.981 [0.981, 0.982] |
| dac_n_quantizers2 | 1.500 | 1.52 [1.51, 1.52] | 0.836 [0.834, 0.837] | 6.99 [6.52, 7.47] | 1.123 [1.119, 1.127] | 18.7 [18.0, 19.3] | 0.754 [0.749, 0.758] | 0.987 [0.987, 0.987] |
| dac_n_quantizers32 | 24.000 | 4.48 [4.47, 4.48] | 0.995 [0.995, 0.995] | 2.35 [2.08, 2.65] | 0.301 [0.300, 0.303] | 5.6 [5.3, 5.9] | 0.944 [0.942, 0.945] | 0.999 [0.999, 0.999] |
| FaCT fact_main (tokens + residual) | 0.423 | n/a | n/a | n/a | 1.268 [1.264, 1.272] | n/a | n/a | 0.962 [0.961, 0.963] |
| FaCT fact_main (tokens only) | 0.279 | n/a | n/a | n/a | 1.706 [1.699, 1.712] | n/a | n/a | 0.878 [0.876, 0.880] |
| FaCT fact_main (tokens only (residual removed)) | 0.279 | n/a | n/a | n/a | 5.104 [5.081, 5.127] | n/a | n/a | 0.136 [0.132, 0.140] |
| FaCT main_c16_rd05 (tokens + residual) | 0.567 | n/a | n/a | n/a | 1.201 [1.197, 1.205] | n/a | n/a | 0.970 [0.969, 0.970] |
| FaCT main_c16_rd05 (tokens only) | 0.279 | n/a | n/a | n/a | 1.701 [1.694, 1.708] | n/a | n/a | 0.881 [0.879, 0.883] |
| FaCT main_c16_rd05 (tokens only (residual removed)) | 0.279 | n/a | n/a | n/a | 1.538 [1.532, 1.543] | n/a | n/a | 0.909 [0.907, 0.910] |
| FaCT main_c8_rd03 (tokens + residual) | 0.423 | n/a | n/a | n/a | 1.282 [1.277, 1.286] | n/a | n/a | 0.960 [0.959, 0.960] |
| FaCT main_c8_rd03 (tokens only) | 0.279 | n/a | n/a | n/a | 1.667 [1.660, 1.673] | n/a | n/a | 0.879 [0.876, 0.881] |
| FaCT main_c8_rd03 (tokens only (residual removed)) | 0.279 | n/a | n/a | n/a | 1.529 [1.524, 1.535] | n/a | n/a | 0.909 [0.907, 0.910] |
| FaCT main_c8_rd05 (tokens + residual) | 0.423 | n/a | n/a | n/a | 1.303 [1.299, 1.308] | n/a | n/a | 0.957 [0.956, 0.958] |
| FaCT main_c8_rd05 (tokens only) | 0.279 | n/a | n/a | n/a | 1.637 [1.631, 1.644] | n/a | n/a | 0.887 [0.885, 0.889] |
| FaCT main_c8_rd05 (tokens only (residual removed)) | 0.279 | n/a | n/a | n/a | 1.519 [1.513, 1.525] | n/a | n/a | 0.913 [0.911, 0.914] |

**Paired mel-L1 difference, FaCT fact_main (tokens only) minus each baseline** (negative = FaCT better; * = 95% CI excludes 0):

| baseline | kbps | Δ mel-L1 |
|---|---|---|
| mimi_n_streams2 | 0.275 | -0.140 [-0.147, -0.133]* |
| mimi_n_streams4 | 0.550 | +0.256 [+0.250, +0.262]* |
| dac_n_quantizers1 | 0.750 | +0.297 [+0.292, +0.303]* |
| xcodec2 | 0.800 | +0.642 [+0.636, +0.649]* |
| mimi_n_streams8 | 1.100 | +0.473 [+0.467, +0.480]* |
| dac_n_quantizers2 | 1.500 | +0.583 [+0.577, +0.588]* |
| dac_n_quantizers32 | 24.000 | +1.404 [+1.398, +1.410]* |
