# Human listening study (DRAFT for discussion, not final)

> **Status:** draft, under discussion with the authors. Not merged to main.
> Becomes binding (fixed before any data is collected) only once agreed.

Objective metrics (UTMOS, WER, speaker similarity, F0) are proxies. Each
paper claim that rests on how speech sounds gets one human test. As in
docs/EVAL_PROTOCOL.md:
- the systems, the utterances and the analysis are fixed here, before anyone
  listens;
- which FaCT rows take part is decided on dev only;
- only test-set utterances are played.

## Claims and tests

| # | Claim | Test | Listeners rate |
|---|---|---|---|
| A | FaCT codec quality vs other low-bitrate codecs | MUSHRA (ITU-R BS.1534-3) | 0–100 quality vs a shown reference |
| B | TTS on FaCT tokens sounds better than on Mimi-2 tokens | MOS + SMOS | naturalness 1–5; speaker similarity to the prompt 1–5 |
| C | Prosody control: resampling FaCT's prosody stream changes intonation and keeps the words | paired judgment | "same words?" yes/no; intonation difference 1–5 |

### A. Codec quality (MUSHRA)

One page per utterance, all conditions shown in random order with blind
labels:

- hidden reference;
- low anchor: the reference low-passed at 3.5 kHz (BS.1534);
- **FaCT, full view (0.28 kbps tokens + ≤0.14 kbps residual):** the dev-selected streaming variant (causal
  encoder, lookahead decoder) and the dev-selected non-causal variant;
- **FaCT, tokens only (0.28 kbps):** the same two variants;
- FocalCodec 25 Hz (0.325 kbps), non-causal;
- FocalCodec-Stream (0.55 kbps), causal;
- Mimi, 2 codebooks (0.275 kbps).

That makes 9 stimuli per page; BS.1534 allows up to 12. FaCT reconstructs
with a speaker vector from a different utterance of the same speaker (the
eval protocol's `spk_ref=other`), as in the tables. Say so next to the
results.

### B. TTS (MOS + SMOS)

- **Systems:** the FaCT LM and the Mimi-2 LM, plus ground-truth recordings as
  an upper anchor, using the same prompt/target pairs as
  `results/tts/test_ls_*`.
- **Naturalness (MOS):** one stimulus per screen, 1–5.
- **Similarity (SMOS):** prompt and generation side by side; "same speaker?"
  on a 1–5 scale.

### C. Prosody control

Pairs from the prosody test (`scripts/eval_tts_prosody.py`), drawn from three
groups:

- FaCT, content fixed: base vs a content-fixed resample;
- FaCT, free: base vs a free resample;
- Mimi-2, first codebook fixed: base vs a resample with codebook 0 fixed.

For each pair the listener answers:
1. "Do the two say exactly the same words?" (yes/no)
2. "How different is the melody/intonation?" (1 = same, 5 = very different)

**Prediction:** FaCT content-fixed pairs get more "same words" answers than
FaCT free pairs and than Mimi-2 fixed pairs, while their intonation-difference
ratings are close to the free pairs'.

## Stimuli

- **Selection:** 40 utterances per test, drawn with a fixed seed from
  LibriSpeech test-clean.
  - Stratified by speaker: at most one utterance per speaker, balanced by sex.
  - 4–10 s long, the same range as the TTS test.
  - Drawn before any output is listened to; the list is saved in the repo.
- **One set for every system:** the same utterances (and for B and C the same
  prompts and seeds as the objective tests), so the human and objective
  numbers are paired.
- **Sample rate:** everything is presented at 16 kHz mono.
  - LibriSpeech is 16 kHz and FocalCodec outputs 16 kHz.
  - FaCT and Mimi are 24 kHz and get resampled down.
  - No system gains from extra bandwidth.
- **Loudness:** every stimulus normalized to −26 LUFS (ITU-R BS.1770), with a
  true-peak limit of −1 dBTP.
- **Emilia:** not used here. Its recordings vary in quality, which confounds
  the codec comparison, and the objective tables already cover it.

## Decisions so far (2026-11-02)

- **Scope:** all three tests (A, B, C).
- **Platform:** a Gradio app hosted as a Hugging Face Space, shared by link,
  not Prolific.
  - Responses go to a private Hugging Face dataset under anonymous ids.
  - Volunteers get one block of about 10–12 min, randomly assigned to A, B or
    C (balanced), with an optional "another block".
  - Inside the app: consent, a background form (native English or not,
    headphones, age band), the Huggins-pitch headphone check and catch trials.
  - Gradio cannot switch conditions at the same playback position, so A is
    reported as "MUSHRA-style".
  - Native-English listeners are the primary analysis group.
  - Fallback if volunteers are too few: link the same app from Prolific with a
    completion code.
- **Ethics:** IRB approval is needed and not yet applied for. No data is
  collected before approval.
- **Timing:** stimuli are generated after the queued decoders finish and the
  dev rule fixes the FaCT rows.
- **Not built yet:** the user asked for planning only.
- **Open:**
  - Who the link goes to (lab, mailing list or social media), which decides
    whether native-English numbers will be enough.
  - Whether the study is English only (assumed yes).

The Prolific-based screening below still applies inside the app, except for
recruitment and pay.

## Listeners and screening

- **Recruitment:** Prolific (superseded: see "Decisions so far").
  - Native English speakers, approval rate ≥ 98%.
  - Desktop only, headphones required.
  - Pay at or above Prolific's recommended hourly rate.
- **Headphone check:** before the test, the Milne et al. (2021) Huggins-pitch
  test; whoever fails is screened out and paid for the screening only.
- **Session:** about 20–25 min per listener, so each test is a separate study.
  - A: 12 MUSHRA pages per listener.
  - B: 40 MOS + 20 SMOS screens.
  - C: 30 pairs.
- **Assignment:** a balanced Latin-square assignment of utterances to
  listeners, so each utterance and condition gets the same number of ratings.
- **Post-screening, fixed now:**
  - A: drop a listener who rates the hidden reference below 90 on more than 15%
    of pages, or the anchor above 90 on more than 15% (BS.1534).
  - B: drop a listener who rates ground truth ≤ 2 on more than 20% of its
    screens.
  - C: 4 identical-file pairs act as catch trials; a listener who answers
    "different words" or intonation ≥ 4 on more than one of them is dropped.
  - Every exclusion is reported with its count.
- **Size:**
  - A: 30 listeners after screening (more than BS.1534's 20, because crowd
    raters are noisier). That is about 9 ratings per condition and utterance.
  - B and C: 40 listeners each after screening.
  - **Power:** a 0.25 MOS difference with a per-rating s.d. of about 1.0 needs
    about 250 paired ratings per contrast; 40 listeners × 40 screens gives
    about 800.

## Analysis (fixed now)

- **MUSHRA and MOS/SMOS:**
  - linear mixed model: score ~ system + (1 | listener) + (1 | utterance);
  - report each system's mean with a 95% CI from a bootstrap that resamples
    listeners and utterances (two-level);
  - pairwise contrasts that the paper's claims need, Holm-corrected:
    - FaCT vs FocalCodec-Stream (streaming);
    - FaCT vs FocalCodec 25 Hz (non-causal);
    - FaCT vs Mimi-2;
    - FaCT LM vs Mimi-2 LM.
- **Prosody test:**
  - "same words": a logistic mixed model on group;
  - intonation difference: the same mixed model as MOS;
  - prediction tested: FaCT fixed vs FaCT free and vs Mimi-2 fixed.
- **Reporting:** all systems and all exclusions are reported, wins and losses
  alike, with the human scores next to the objective ones (UTMOS, WER, SIM) for
  the same utterances.

## Ethics and data

- Ethics approval from the institution before recruiting (IIIT-Delhi IRB or
  equivalent).
- Informed consent; no personal data stored beyond the Prolific id; data
  stored anonymized.
- LibriSpeech is CC BY 4.0, so playing it in a research study is fine.

## Still to build

Each needs about 1 GPU-hour at most and nothing else running, so it can wait
for the CVPR GPU crunch to pass.

1. `scripts/listening/select_utterances.py`: the fixed-seed stimulus list.
2. A stimulus-generation job that decodes and saves the audio for every system,
   with loudness normalization and the anchor.
3. A webMUSHRA config generator, hosted for Prolific, covering MUSHRA, MOS/SMOS
   and the pair test.
4. `scripts/listening/analyze.py`: post-screening, mixed models, bootstrap CIs
   and the paper table.

**Timeline:** about a week to build and pilot (5 listeners, then fix wording),
then 2–3 days of data collection.

## Decisions for the authors

- Platform: Prolific (assumed above) or in-lab listeners.
- Ethics approval: does the institution require it, and is one already in
  place?
- Budget. Rough figure: about 110 listeners × about 25 min at Prolific's
  recommended rate plus its fee, roughly £600–700.
- Whether to also run A on Emilia test.
