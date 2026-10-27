#!/bin/bash
# LOGIN NODE (needs internet), once: download the LibriTTS training subsets
# (OpenSLR 60, CC BY 4.0) - FocalCodec's training corpus - to SCRATCH for the
# decoder runs. Resumable (wget -c); download only, extraction and the
# manifest run as a compute job (slurm/prepare_libritts.sbatch).
#   source /leonardo_work/AIFAC_S07_041/fact/fact_env.sh
#   bash scripts/fetch_libritts.sh            # in tmux: ~78 GB
# Only train-* subsets: LibriTTS splits follow LibriSpeech's, so no
# LibriSpeech test-clean speaker is in them.
set -euo pipefail
: "${FACT_DATA:?source fact_env.sh first}"
case "$FACT_DATA" in /leonardo_scratch/*) ;; *) echo "FACT_DATA=$FACT_DATA is not on SCRATCH"; exit 1;; esac
DEST=$FACT_DATA/data/raw/libritts/tars
mkdir -p "$DEST"
MIRRORS="https://openslr.elda.org/resources/60 https://www.openslr.org/resources/60 https://us.openslr.org/resources/60"
for f in train-clean-100.tar.gz train-clean-360.tar.gz train-other-500.tar.gz; do
    if [ -f "$DEST/$f.done" ]; then echo "have $f"; continue; fi
    ok=""
    for m in $MIRRORS; do
        echo "== $f from $m"
        if nice -n 19 wget -c --tries=20 --timeout=60 -q --show-progress -O "$DEST/$f" "$m/$f"; then ok=1; break; fi
    done
    [ -n "$ok" ] || { echo "download of $f failed on every mirror"; exit 1; }
    stat -c %s "$DEST/$f" > "$DEST/$f.done"
done
ls -l "$DEST"
echo "Downloads complete. Tell Claude 'LibriTTS downloaded' - the extraction + manifest job is then queued."
