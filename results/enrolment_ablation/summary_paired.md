# Enrolment ablation (test sets; paired 95% bootstrap CIs, * = CI excludes 0)

- other: enrolment = another utterance of the same speaker (every reported FaCT row).
- mean: one fixed speaker vector (the eval set's mean direction) for all: no enrolment.
- xspk: enrolment from a different speaker.

## LibriSpeech test-clean: disjref, full (n=2620)

| enrolment | WER % | UTMOS | SIM | ΔWER % vs other | ΔUTMOS vs other | ΔSIM vs other |
|---|---|---|---|---|---|---|
| other | 3.73 | 3.474 | 0.922 |  |  |  |
| mean | 3.46 | 3.519 | 0.826 | -0.27 [-0.48, -0.09]* | +0.045 [+0.033, +0.057]* | -0.096 [-0.099, -0.092]* |
| xspk | 3.64 | 3.308 | 0.758 | -0.09 [-0.31, +0.11] | -0.167 [-0.185, -0.148]* | -0.164 [-0.169, -0.159]* |

No enrolment vs FocalCodec 25 Hz (neither has side information), same utterances: ΔWER % +0.08 [-0.33, +0.41], ΔUTMOS -0.622 [-0.638, -0.606]*, ΔSIM -0.129 [-0.132, -0.126]*

## LibriSpeech test-clean: disjref, discrete_only (n=2620)

| enrolment | WER % | UTMOS | SIM | ΔWER % vs other | ΔUTMOS vs other | ΔSIM vs other |
|---|---|---|---|---|---|---|
| other | 4.54 | 3.417 | 0.910 |  |  |  |
| mean | 4.30 | 3.482 | 0.806 | -0.24 [-0.84, +0.12] | +0.065 [+0.052, +0.078]* | -0.104 [-0.108, -0.101]* |
| xspk | 4.50 | 3.231 | 0.734 | -0.04 [-0.61, +0.33] | -0.187 [-0.207, -0.167]* | -0.177 [-0.182, -0.171]* |

No enrolment vs FocalCodec 25 Hz (neither has side information), same utterances: ΔWER % +0.92 [+0.52, +1.29]*, ΔUTMOS -0.659 [-0.677, -0.642]*, ΔSIM -0.149 [-0.153, -0.146]*

## LibriSpeech test-clean: la4, full (n=2620)

| enrolment | WER % | UTMOS | SIM | ΔWER % vs other | ΔUTMOS vs other | ΔSIM vs other |
|---|---|---|---|---|---|---|
| other | 3.77 | 3.248 | 0.932 |  |  |  |
| mean | 3.74 | 3.403 | 0.830 | -0.03 [-0.20, +0.15] | +0.155 [+0.143, +0.168]* | -0.102 [-0.105, -0.099]* |
| xspk | 3.85 | 3.051 | 0.745 | +0.07 [-0.10, +0.23] | -0.197 [-0.218, -0.177]* | -0.187 [-0.192, -0.181]* |

No enrolment vs FocalCodec-Stream (neither has side information), same utterances: ΔWER % +0.07 [-0.32, +0.41], ΔUTMOS -0.472 [-0.488, -0.456]*, ΔSIM -0.121 [-0.124, -0.118]*

## LibriSpeech test-clean: la4, discrete_only (n=2620)

| enrolment | WER % | UTMOS | SIM | ΔWER % vs other | ΔUTMOS vs other | ΔSIM vs other |
|---|---|---|---|---|---|---|
| other | 4.66 | 3.124 | 0.921 |  |  |  |
| mean | 4.67 | 3.307 | 0.816 | +0.01 [-0.18, +0.22] | +0.183 [+0.170, +0.196]* | -0.105 [-0.110, -0.102]* |
| xspk | 4.81 | 2.909 | 0.725 | +0.15 [-0.06, +0.37] | -0.215 [-0.237, -0.193]* | -0.196 [-0.202, -0.190]* |

No enrolment vs FocalCodec-Stream (neither has side information), same utterances: ΔWER % +1.00 [+0.59, +1.39]*, ΔUTMOS -0.569 [-0.585, -0.552]*, ΔSIM -0.135 [-0.139, -0.132]*

## Emilia test: disjref, full (n=3783)

| enrolment | WER % | UTMOS | SIM | ΔWER % vs other | ΔUTMOS vs other | ΔSIM vs other |
|---|---|---|---|---|---|---|
| other | 9.21 | 3.271 | 0.949 |  |  |  |
| mean | 9.29 | 3.278 | 0.807 | +0.07 [-0.20, +0.34] | +0.007 [-0.003, +0.017] | -0.142 [-0.146, -0.138]* |
| xspk | 10.23 | 3.111 | 0.743 | +1.02 [+0.72, +1.32]* | -0.160 [-0.174, -0.145]* | -0.206 [-0.212, -0.201]* |

No enrolment vs FocalCodec 25 Hz (neither has side information), same utterances: ΔWER % +1.41 [+1.16, +1.69]*, ΔUTMOS -0.473 [-0.487, -0.459]*, ΔSIM -0.150 [-0.154, -0.146]*

## Emilia test: la4, full (n=3783)

| enrolment | WER % | UTMOS | SIM | ΔWER % vs other | ΔUTMOS vs other | ΔSIM vs other |
|---|---|---|---|---|---|---|
| other | 9.70 | 3.031 | 0.952 |  |  |  |
| mean | 9.74 | 3.232 | 0.801 | +0.04 [-0.25, +0.32] | +0.201 [+0.191, +0.212]* | -0.151 [-0.155, -0.147]* |
| xspk | 10.84 | 2.801 | 0.726 | +1.15 [+0.82, +1.46]* | -0.229 [-0.246, -0.212]* | -0.226 [-0.231, -0.220]* |

No enrolment vs FocalCodec-Stream (neither has side information), same utterances: ΔWER % -0.02 [-0.35, +0.29], ΔUTMOS -0.071 [-0.085, -0.056]*, ΔSIM -0.151 [-0.156, -0.147]*

