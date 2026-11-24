# Staged requests (prepared 2026-11-24, start after the Leonardo maintenance)

These are complete, validated queue requests kept OUT of `ops/queue/`, so the
runner does not submit them. Start a group by moving its files, with the
user's go-ahead:

```bash
git mv ops/staged/2026-11-26-facodec-h2h-*.json ops/queue/   # example: group A
python -m fact.ops.queue validate --repo .
git commit -m "queue group A" && git push
```

All 61 pass `fact.ops.queue validate` (checked in a copy of the repo).

| Group | Files | What | Compute | Weakness addressed |
|---|---|---|---|---|
| A | `facodec-h2h-{ls,emilia}-shard0..7` | FACodec under FaCT's factorization + voice-conversion tests, same metric names (scripts/eval_facodec.py) | 16 x 1 GPU, about 1-3 h each | no factorized baseline |
| B | `recon-{ls,emilia}-{facodec,facodec_no_residual,zipcodec}` | reconstruction rows in the standard tables | 6 x 1 GPU, a few h | FACodec / ZipCodec missing |
| C | `spkfull-{ls,emilia}-{la4,disjref}` | speaker reference = the utterance itself (upper bound of the enrolment protocol) | 4 x 1 GPU | speaker similarity |
| D | `rtf-fact`, `rtf-baselines` | real-time factor on GPU and CPU; with the existing la0/la4/la8 rows gives the latency curve | 2 x 1 GPU, < 4 h | streaming evidence |
| E | `train-abl_{single_fsq,vae}-p1..5`, `train-wavedec_abl_*-p1..3`, `eval-{dev,ls,emilia}-abl_*` | matched ablations of main_c8_rd05: factorization off (single FSQ, same bits) and discreteness off (continuous latent, same total capacity 23.45 nats/frame); `--ablation-of` refuses any change outside the bottleneck | 2 x (5 x 2-node days + 3 x 1-node days) + evals | method contribution unproven |
| F | `tts-tok-focal25`, `tts-train-focal25-p1..3`, `tts-dev-focal25-*` | same 300M TTS LM on FocalCodec 25 Hz tokens, same utterances (requires the fact_b1 and mimi2_b1 token dirs), compute-matched (same steps and tokens per batch; 25 Hz needs --max-frames 800); test requests are written after the dev selection | 64-task array + 3 x 2-node days + 4 evals | TTS has one baseline |

**Order:** A, B, C, D first (cheap, single GPU). E and F are the long runs:
start them together if nodes allow.

**SCRATCH risk:** E reads `shards/train_v2`, F reads the train shards, the
audio index and the Emilia audio, and F needs the existing
`tts_tokens/{fact_b1,mimi2_b1}`, all on SCRATCH (purge about 40 days after
creation or last access, early December for the October files). Check the
purge dates of those directories before launching E/F, and launch F's
tokenization first.
