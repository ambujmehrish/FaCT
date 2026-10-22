## Emilia-EN test (5,000 utterances, speaker-disjoint)

Mean [95% bootstrap CI] over utterances (n = 5000). WER is corpus-level vs the ground-truth transcript. mel-L1 uses bins below the source audio's Nyquist. FaCT rows are pre-vocoder (mel domain): waveform metrics are n/a until the stage-C vocoder exists.

| system | kbps | PESQ-WB ↑ | STOI ↑ | WER % ↓ | mel-L1 ↓ | F0 RMSE ↓ | V/UV F1 ↑ | energy r ↑ |
|---|---|---|---|---|---|---|---|---|
| mimi_n_streams2 | 0.275 | 1.24 [1.24, 1.24] | 0.770 [0.769, 0.771] | 16.40 [15.86, 16.93] | 1.838 [1.833, 1.843] | 28.5 [28.0, 29.0] | 0.726 [0.723, 0.729] | 0.956 [0.956, 0.957] |
| mimi_n_streams4 | 0.550 | 1.61 [1.61, 1.62] | 0.852 [0.851, 0.853] | 11.19 [10.73, 11.68] | 1.497 [1.493, 1.500] | 16.7 [16.3, 17.1] | 0.796 [0.793, 0.799] | 0.973 [0.973, 0.973] |
| dac_n_quantizers1 | 0.750 | 1.18 [1.18, 1.18] | 0.752 [0.750, 0.753] | 33.18 [32.34, 34.00] | 1.514 [1.509, 1.518] | 30.3 [29.7, 30.9] | 0.630 [0.626, 0.634] | 0.970 [0.969, 0.970] |
| xcodec2 | 0.800 | 2.01 [2.00, 2.02] | 0.900 [0.899, 0.900] | 7.42 [7.01, 7.86] | 1.063 [1.060, 1.067] | 14.4 [14.0, 14.7] | 0.829 [0.827, 0.831] | 0.991 [0.991, 0.991] |
| mimi_n_streams8 | 1.100 | 2.18 [2.18, 2.19] | 0.905 [0.904, 0.906] | 7.70 [7.32, 8.10] | 1.300 [1.297, 1.303] | 13.3 [12.9, 13.7] | 0.836 [0.834, 0.838] | 0.980 [0.979, 0.980] |
| dac_n_quantizers2 | 1.500 | 1.52 [1.51, 1.52] | 0.839 [0.838, 0.840] | 13.20 [12.70, 13.74] | 1.227 [1.223, 1.231] | 19.6 [19.1, 20.1] | 0.775 [0.772, 0.778] | 0.988 [0.988, 0.988] |
| dac_n_quantizers32 | 24.000 | 4.33 [4.33, 4.34] | 0.995 [0.994, 0.995] | 5.74 [5.38, 6.14] | 0.416 [0.414, 0.417] | 5.7 [5.5, 5.9] | 0.943 [0.942, 0.944] | 0.999 [0.998, 0.999] |
| FaCT fact_main (tokens + residual) | 0.423 | n/a | n/a | n/a | 1.347 [1.343, 1.352] | n/a | n/a | 0.965 [0.965, 0.966] |
| FaCT fact_main (tokens only) | 0.279 | n/a | n/a | n/a | 1.851 [1.846, 1.857] | n/a | n/a | 0.880 [0.878, 0.881] |
| FaCT fact_main (tokens only (residual removed)) | 0.279 | n/a | n/a | n/a | 4.938 [4.922, 4.957] | n/a | n/a | 0.123 [0.121, 0.125] |
| FaCT main_c16_rd05 (tokens + residual) | 0.567 | n/a | n/a | n/a | 1.279 [1.274, 1.283] | n/a | n/a | 0.973 [0.973, 0.974] |
| FaCT main_c16_rd05 (tokens only) | 0.279 | n/a | n/a | n/a | 1.882 [1.876, 1.888] | n/a | n/a | 0.879 [0.877, 0.881] |
| FaCT main_c16_rd05 (tokens only (residual removed)) | 0.279 | n/a | n/a | n/a | 1.634 [1.628, 1.640] | n/a | n/a | 0.916 [0.915, 0.917] |
| FaCT main_c8_rd03 (tokens + residual) | 0.423 | n/a | n/a | n/a | 1.364 [1.359, 1.368] | n/a | n/a | 0.964 [0.963, 0.964] |
| FaCT main_c8_rd03 (tokens only) | 0.279 | n/a | n/a | n/a | 1.801 [1.795, 1.808] | n/a | n/a | 0.885 [0.883, 0.886] |
| FaCT main_c8_rd03 (tokens only (residual removed)) | 0.279 | n/a | n/a | n/a | 1.627 [1.621, 1.633] | n/a | n/a | 0.917 [0.915, 0.918] |
| FaCT main_c8_rd05 (tokens + residual) | 0.423 | n/a | n/a | n/a | 1.385 [1.380, 1.390] | n/a | n/a | 0.962 [0.961, 0.962] |
| FaCT main_c8_rd05 (tokens only) | 0.279 | n/a | n/a | n/a | 1.770 [1.763, 1.776] | n/a | n/a | 0.890 [0.888, 0.891] |
| FaCT main_c8_rd05 (tokens only (residual removed)) | 0.279 | n/a | n/a | n/a | 1.614 [1.607, 1.619] | n/a | n/a | 0.919 [0.918, 0.920] |

**Paired mel-L1 difference, FaCT fact_main (tokens only) minus each baseline** (negative = FaCT better; * = 95% CI excludes 0):

| baseline | kbps | Δ mel-L1 |
|---|---|---|
| mimi_n_streams2 | 0.275 | +0.013 [+0.006, +0.019]* |
| mimi_n_streams4 | 0.550 | +0.355 [+0.349, +0.361]* |
| dac_n_quantizers1 | 0.750 | +0.338 [+0.332, +0.344]* |
| xcodec2 | 0.800 | +0.788 [+0.783, +0.793]* |
| mimi_n_streams8 | 1.100 | +0.551 [+0.546, +0.557]* |
| dac_n_quantizers2 | 1.500 | +0.624 [+0.619, +0.630]* |
| dac_n_quantizers32 | 24.000 | +1.436 [+1.430, +1.442]* |
