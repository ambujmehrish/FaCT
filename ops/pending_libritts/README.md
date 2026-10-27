Requests waiting for the LibriTTS download (scripts/fetch_libritts.sh, login
node). Not read by the runner. Once the download is done, move them into
ops/queue/ (git mv), validate, commit and push: extraction + manifest, then
three stage-D decoders trained on LibriTTS (non-causal, +80 ms, +160 ms
lookahead) with dev / LibriSpeech / Emilia evals.
