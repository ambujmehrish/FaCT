#!/bin/bash
# Extract the LibriTTS training tarballs (scripts/fetch_libritts.sh) on SCRATCH
# and build the decoder-training manifest $FACT_DATA/manifests/libritts_train.tsv.
# Refuses if a tarball is missing or the utterance count is far from LibriTTS's.
#SBATCH --job-name=fact-prep-libritts
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                    # smallest allocation on the GPU partition
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
source "${FACT_ENV:-$WORK/fact/fact_env.sh}"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1
ROOT=$FACT_DATA/data/raw/libritts
for f in train-clean-100 train-clean-360 train-other-500; do
    [ -f "$ROOT/tars/$f.tar.gz.done" ] || { echo "missing $ROOT/tars/$f.tar.gz (run scripts/fetch_libritts.sh)"; exit 1; }
    if [ -f "$ROOT/$f.extracted" ]; then echo "have $f"; continue; fi
    tar -xzf "$ROOT/tars/$f.tar.gz" -C "$ROOT"      # a corrupt archive fails here
    touch "$ROOT/$f.extracted"
done
OUT=$FACT_DATA/manifests/libritts_train.tsv
python -m fact.data.manifest libritts --root "$ROOT/LibriTTS" --out "$OUT.partial"
python - "$OUT.partial" <<'PY'
import sys
# LibriTTS (Zen et al. 2019): 33,236 + 116,500 + 205,044 = 354,780 train utterances.
n = sum(1 for ln in open(sys.argv[1], encoding="utf-8") if ln.strip())
subsets = {s: sum(1 for ln in open(sys.argv[1], encoding="utf-8") if f"/{s}/" in ln)
           for s in ("train-clean-100", "train-clean-360", "train-other-500")}
print(f"{n} utterances {subsets}")
if n < 0.99 * 354_780 or any(c == 0 for c in subsets.values()):
    sys.exit(f"unexpected LibriTTS manifest: {n} utterances {subsets}")
PY
mv "$OUT.partial" "$OUT"
echo "wrote $OUT"
