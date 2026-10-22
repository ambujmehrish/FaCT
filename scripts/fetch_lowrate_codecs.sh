#!/bin/bash
# LOGIN NODE (needs internet), once: source code + weights of the dedicated
# low-bitrate codecs (fact/baselines/lowrate.py) for offline compute jobs.
#   source /leonardo_work/AIFAC_S07_041/fact/fact_env.sh
#   bash scripts/fetch_lowrate_codecs.sh
# Code: pinned commits cloned into $FACT_CACHE/codecs/<name> (no pip install,
# nothing in the venv changes). Weights: Hugging Face cache ($HF_HOME), only
# the files inference needs (~5.5 GB).
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
             "lucadellalib/focalcodec_12_5hz", "lucadellalib/focalcodec_50hz_4k_causal"]:
    print(repo, snapshot_download(repo, allow_patterns=["config.json", "model.safetensors"]))
print(hf_hub_download("Alethia/BigCodec", "bigcodec.pt"))
for repo, fname, _ in WAVTOKENIZER.values():
    print(hf_hub_download(repo, fname))
PY
echo "Done. Check (login node is fine, CPU, ~1 min each):"
echo "  HF_HUB_OFFLINE=1 python -c \"from fact.baselines import base; print(base.build('bigcodec', device='cpu').discrete_bps)\""
