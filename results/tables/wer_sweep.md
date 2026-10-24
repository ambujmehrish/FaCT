## Dev WER sweep (Emilia dev, vocoded, other-utterance speaker reference)

```
main_c16_rd05@0400000: 9 settings, 1158 common dev utterances; mel-L1 limit 1.184
  T     CFG   WER %   mel-L1  UTMOS
  0.0   1.5    10.39  1.120   3.38
  0.0   2.0    10.09  1.139   3.37
  0.0   3.0    10.04  1.207   3.28 (mel-L1 over limit)
  0.3   1.5    10.41  1.134   3.36
  0.3   2.0     9.97  1.152   3.35 <- selected
  0.3   3.0     9.92  1.220   3.27 (mel-L1 over limit)
  0.5   1.5    10.70  1.164   3.31
  0.5   2.0    10.34  1.181   3.31
  0.5   3.0    10.25  1.250   3.24 (mel-L1 over limit)
0.3 32 2.0
```

```
main_c8_rd03@0400000: 9 settings, 1158 common dev utterances; mel-L1 limit 1.295
  T     CFG   WER %   mel-L1  UTMOS
  0.0   1.5    13.29  1.228   3.31
  0.0   2.0    13.00  1.247   3.31 <- selected
  0.0   3.0    13.00  1.314   3.24 (mel-L1 over limit)
  0.3   1.5    13.68  1.243   3.29
  0.3   2.0    13.35  1.260   3.29
  0.3   3.0    13.11  1.329   3.21 (mel-L1 over limit)
  0.5   1.5    13.95  1.275   3.23
  0.5   2.0    13.77  1.293   3.24
  0.5   3.0    13.30  1.365   3.16 (mel-L1 over limit)
0.0 32 2.0
```

```
main_c8_rd05@0400000: 9 settings, 1158 common dev utterances; mel-L1 limit 1.374
  T     CFG   WER %   mel-L1  UTMOS
  0.0   1.5    15.09  1.306   3.23
  0.0   2.0    14.94  1.329   3.23
  0.0   3.0    15.12  1.407   3.14 (mel-L1 over limit)
  0.3   1.5    15.03  1.321   3.21
  0.3   2.0    14.92  1.342   3.20 <- selected
  0.3   3.0    15.31  1.420   3.12 (mel-L1 over limit)
  0.5   1.5    16.00  1.354   3.15
  0.5   2.0    15.68  1.374   3.15 (mel-L1 over limit)
  0.5   3.0    15.64  1.450   3.08 (mel-L1 over limit)
0.3 32 2.0
```

