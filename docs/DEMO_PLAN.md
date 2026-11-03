# Streaming demo plan (agreed 2026-11-02; not built yet)

**Decision:**
- **A**, a demo web page with pre-rendered samples;
- **C**, a short screen recording of a live demo running on a local GPU;
- streaming variant: **la4** (causal encoder; decoder lookahead 4 frames of convolution
  + 2 frames of iSTFT overlap = 120 ms; about 200 ms algorithmic latency end to end);
- includes a **prosody slider**.

There is no hosted live app, because of GPU cost and because network latency
would hide the model's own latency.

## A. Demo page (pre-rendered, shareable, permanent)

Audio comes from LibriSpeech test-clean, at 16 kHz and loudness-matched. It
shares its stimuli with the listening study (docs/LISTENING_STUDY.md) where it
can.

1. **Streaming = offline:** the same utterance encoded and decoded chunk by
   chunk (80 ms) and in one pass, with the audio for both and the token
   agreement (expected identical up to float rounding; also a unit test).
2. **Side by side:** original vs FaCT la4 full (0.28 kbps tokens + ≤0.14 kbps residual) vs FaCT la4 tokens
   only (0.28 kbps) vs FocalCodec-Stream (0.55 kbps) vs Mimi-2 (0.275 kbps),
   each labelled with bitrate and algorithmic latency.
3. **Prosody control at the codec level:**
   - ±4 semitones through the prosody tokens, and the same shift through the
     content tokens, which leaves pitch unchanged;
   - flattened prosody;
   - prosody swapped from another utterance.
4. **Prosody control in TTS:** a base sample, content-fixed prosody resamples
   (same words, new intonation) and free resamples.
5. **Streaming TTS:** text in, audio out frame by frame, with time to first
   audio.
6. **Latency table:**

   Worst-case algorithmic latency, from `fact/eval/latency.py` (checked against
   a perturbation measurement in `tests/test_latency.py`; corrected
   2026-11-03, the earlier "about 181 ms" missed the iSTFT term):

   | Component | ms |
   |---|---|
   | Token granularity (a token waits for the rest of its 4 frames) | 60 |
   | Mel analysis lookahead (centred STFT, half the 960-sample window at 24 kHz) | 20 |
   | la4 decoder convolution lookahead (4 frames) | 80 |
   | iSTFT overlap-add in the decoder (2 frames) | 40 |
   | **Total, la4, before compute** | **200** |

   la0 (fully causal decoder) is 120 ms and la8 is 280 ms. Add measured
   compute per frame on GPU. FocalCodec-Stream's figure must be stated on the
   same definition (worst-case input-to-output algorithmic latency), not as
   its frame or lookahead length alone.

## C. Screen recording (local GPU)

- **Setup:** about 3 s of enrolment speech for the speaker vector, then live
  speech in and FaCT la4 speech out.
- **On screen:** a running latency readout (algorithmic and compute,
  separately) and the token stream.
- **Prosody slider while speaking:**
  - a pitch shift and an intonation-flattening amount, both applied to the
    prosody tokens (or to the F0 that the prosody tokens come from);
  - words stay the same;
  - the slider maps to the same operations as section A.3, so the video shows
    what the paper measures.

## Engineering

- **Streaming encoder:**
  - incremental inference with a key-value cache, one token per 4 mel frames;
  - a streaming mel front end that accounts for the STFT lookahead;
  - for the la4 model the encoder stays strictly causal.
- **Streaming decoder:** re-run the la4 generator on a sliding window that
  covers its receptive field and keep the new samples. This is exact for
  bounded convolutions; the overlap-add at the window edges needs care.
- **Test:** chunked tokens and audio equal offline (fixed tolerance), on the
  tiny config and on the real la4 checkpoint.
- **Measured, not assumed:** compute per 80 ms frame on GPU (and CPU) for the
  la4 model.
- **Speaker vector:** from an enrolment clip, matching the evaluation's
  `spk_ref=other` setting.

## Order of work

1. Streaming encoder and decoder, plus the equivalence test (no GPU; tiny
   config).
2. Check equivalence and real-time speed on the real checkpoint (minutes on
   one GPU).
3. Render the demo-page audio (about 1 GPU-hour, together with the listening
   study's stimuli, after the decoders finish and the dev rule fixes the FaCT
   rows).
4. Build the page (static HTML).
5. Record the live demo locally.
