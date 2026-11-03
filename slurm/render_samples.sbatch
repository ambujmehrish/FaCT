#!/bin/bash
# Listening samples: codec reconstructions, zero-shot TTS and prosody-control
# samples for the same N LibriSpeech test-clean utterances, plus index.html.
# One GPU. Account: IscrC_RHYME_0 (approved only for this script; see
# fact/ops/queue.py ACCOUNT_SCRIPTS).
#   OUT=<dir> [N=8] sbatch slurm/render_samples.sbatch
#SBATCH --job-name=fact-samples
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
if [ -n "${FACT_MODULES:-}" ]; then module purge; module load $FACT_MODULES; fi
source "${FACT_ENV:-$WORK/fact/fact_env.sh}"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

OUT=${OUT:?set OUT=<samples dir>}
N=${N:-8}
R=${FACT_RUNS:-$FACT_BASE/runs}
LS=${EVAL_DIR:-$FACT_BASE/data/eval/librispeech_test_clean_full}
SPK=spk_ref=other

python scripts/render_codec_samples.py --eval-dir "$LS" --n "$N" --out "$OUT" --device cuda \
    --spec "fact_wave:decoder=$R/wavedec_main_c8_rd05_la4_ltts/step_0250000.pt:view=full:$SPK:label=FaCT_la4-200ms_LibriTTS_0.28+0.14kbps" \
    --spec "fact_wave:decoder=$R/wavedec_main_c8_rd05_la4_ltts/step_0250000.pt:view=discrete_only:$SPK:label=FaCT_la4-200ms_LibriTTS_0.28kbps" \
    --spec "fact_wave:decoder=$R/wavedec_main_c8_rd05/step_0250000.pt:view=full:$SPK:label=FaCT_noncausal-decoder_Emilia_0.28+0.14kbps" \
    --spec "fact:ckpt=$R/ssl_nc_c8_rd05/step_0400000.pt:vocoder=$R/vocoder/step_0250000.pt:view=full:nfe=32:temperature=0.3:cfg_scale=2.0:$SPK:label=FaCT_ssl_nc-flow_0.28+0.14kbps" \
    --spec "focalcodec:config=lucadellalib/focalcodec_25hz:label=FocalCodec_25Hz_0.325kbps" \
    --spec "focalcodec:config=lucadellalib/focalcodec_50hz_2k_causal:label=FocalCodec-Stream_0.55kbps" \
    --spec "mimi:n_streams=2:label=Mimi-2_0.275kbps"

# TTS and prosody control: the same LMs, decoder and dev-selected sampling as
# the test runs; per-utterance seeds make these the test runs' generations.
SAMP="--temperature 0.8 --top-k 50"
python scripts/eval_tts.py --lm "$R/tts_fact_b1r/step_0200000.pt" \
    --wave-decoder "$R/wavedec_main_c8_rd05/step_0250000.pt" --eval-dir "$LS" \
    --limit-targets "$N" --no-quality $SAMP --device cuda \
    --out "$OUT/tts/fact.json" --save-audio "$OUT/tts/FaCT-LM"
python scripts/eval_tts.py --lm "$R/tts_mimi2_b1r/step_0200000.pt" --eval-dir "$LS" \
    --limit-targets "$N" --no-quality $SAMP --device cuda \
    --out "$OUT/tts/mimi2.json" --save-audio "$OUT/tts/Mimi-2-LM"
python scripts/eval_tts_prosody.py --lm "$R/tts_fact_b1r/step_0200000.pt" \
    --wave-decoder "$R/wavedec_main_c8_rd05/step_0250000.pt" --eval-dir "$LS" \
    --limit-targets "$N" --resamples 3 $SAMP --device cuda \
    --out "$OUT/prosody/fact.json" --save-audio "$OUT/prosody/FaCT-LM"
python scripts/eval_tts_prosody.py --lm "$R/tts_mimi2_b1r/step_0200000.pt" --eval-dir "$LS" \
    --limit-targets "$N" --resamples 3 $SAMP --device cuda \
    --out "$OUT/prosody/mimi2.json" --save-audio "$OUT/prosody/Mimi-2-LM"

python scripts/make_sample_page.py --dir "$OUT"
echo "samples: $OUT/index.html"
