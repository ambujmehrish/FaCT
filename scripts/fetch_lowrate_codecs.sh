#!/bin/bash
# LOGIN NODE (needs internet), once: source code + weights of the dedicated
# low-bitrate codecs (fact/baselines/lowrate.py) and the UTMOS / speaker-
# similarity models (fact/eval/quality.py) for offline compute jobs.
#   source /leonardo_work/AIFAC_S07_041/fact/fact_env.sh
#   bash scripts/fetch_lowrate_codecs.sh
# Code: pinned commits cloned into $FACT_CACHE/codecs/<name> (no pip install,
# nothing in the venv changes). Weights: Hugging Face cache ($HF_HOME), only
# the files inference needs (~6.3 GB).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${FACT_CACHE:?source fact_env.sh first}"
DEST=$FACT_CACHE/codecs
mkdir -p "$DEST"
python - "$DEST" <<'PY'
import subprocess, sys
from pathlib import Path
from fact.baselines.lowrate import REPOS
dest = Path(sys.argv[1])
for name, (url, commit) in REPOS.items():
    d = dest / name
    if not d.exists():
        subprocess.run(["git", "clone", "-q", url, str(d)], check=True)
    subprocess.run(["git", "-C", str(d), "fetch", "-q", "origin"], check=False)
    subprocess.run(["git", "-C", str(d), "checkout", "-q", commit], check=True)
    head = subprocess.run(["git", "-C", str(d), "rev-parse", "--short", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    print(f"{name}: {head} ({d})")
PY
python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
from fact.baselines.lowrate import WAVTOKENIZER
for repo in ["lucadellalib/focalcodec_50hz", "lucadellalib/focalcodec_25hz",
             "lucadellalib/focalcodec_12_5hz", "lucadellalib/focalcodec_50hz_4k_causal",
             "lucadellalib/focalcodec_50hz_2k_causal", "lucadellalib/focalcodec_50hz_65k_causal"]:
    print(repo, snapshot_download(repo, allow_patterns=["config.json", "model.safetensors"]))
print(hf_hub_download("Alethia/BigCodec", "bigcodec.pt"))
for repo, fname, _ in WAVTOKENIZER.values():
    print(hf_hub_download(repo, fname))
# EnCodec 24 kHz (causal baseline, fact/baselines/encodec.py).
print(snapshot_download("facebook/encodec_24khz"))
# WavLM-large: frozen SSL encoder input of the #4 run (fact/modules/ssl.py).
print(snapshot_download("microsoft/wavlm-large", allow_patterns=["*.json", "*.bin", "*.safetensors"]))
# Quality metrics (fact/eval/quality.py): speaker-verification WavLM.
from fact.eval.quality import SV_MODEL
print(snapshot_download(SV_MODEL))
PY
# UTMOS22 weights (GitHub release of the pinned SpeechMOS port).
W=$DEST/SpeechMOS/utmos22_strong_step7459_v1.pt
[ -s "$W" ] || curl -fsSL -o "$W.tmp" \
    https://github.com/tarepan/SpeechMOS/releases/download/v1.0.0/utmos22_strong_step7459_v1.pt
[ -s "$W" ] || mv "$W.tmp" "$W"
ls -lh "$W"
echo "Done. Check (login node is fine, CPU, ~1 min each):"
echo "  HF_HUB_OFFLINE=1 python -c \"from fact.baselines import base; print(base.build('bigcodec', device='cpu').discrete_bps)\""
