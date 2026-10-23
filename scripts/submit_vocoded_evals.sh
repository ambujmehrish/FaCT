#!/bin/bash
# Waveform evaluation of FaCT through the stage-C vocoder (and, with WAVE=,
# through the stage-D waveform decoder), on both full test sets, with every
# metric the baselines have (PESQ/STOI/WER/F0/mel/UTMOS/SIM). Results land
# next to the baselines (results/baseline_recon_{ls,emilia}/<label>.json),
# so the recon tables show FaCT and baselines in the same columns.
#
#   bash scripts/submit_vocoded_evals.sh                       # vocoder rows
#   WAVE=<runs>/wavedec_main_c8_rd05/step_0250000.pt WAVE_RUN=main_c8_rd05 \
#     bash scripts/submit_vocoded_evals.sh                     # + stage-D rows
#
# Sampling: the dev-sweep winner (scripts/submit_dev_sweep.sh, rule fixed in
# advance: lowest dev full-view mel-L1 of SWEEP_RUN) unless T/NFE/G are set;
# SELECT=wer uses the dev WER sweep's winner instead (see below).
# Skips outputs that exist and jobs already queued.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
D=${D:-/leonardo_scratch/large/userexternal/amehrish/fact}
ACCOUNT=${ACCOUNT:-AIFAC_S07_041}
RUNS=${RUNS:-"fact_main main_c8_rd05 main_c8_rd03 main_c16_rd05"}
STEP=${STEP:-0400000}
# Non-final checkpoints get the step in their labels (fine-tunes end at 100K).
STEP_TAG=$([ "$STEP" = 0400000 ] && echo "" || echo "@$STEP")
LS=${LS:-/leonardo_work/AIFAC_S07_041/fact/data/eval/librispeech_test_clean_full}
EM=${EM:-$D/manifests/test.tsv}
VOC=${VOC:-$(ls "$D"/runs/vocoder/step_*.pt 2>/dev/null | sort | tail -1)}
[ -f "$VOC" ] || { echo "no vocoder checkpoint (set VOC=)"; exit 1; }
SWEEP_RUN=${SWEEP_RUN:-main_c8_rd05}
# SELECT=wer: take the setting from the dev WER sweep instead
# (scripts/submit_wer_sweep.sh + select_wer_setting.py, rule fixed in advance);
# labels then end in _werSel so the mel-L1-selected results are kept.
SUFFIX=""
if [ "${SELECT:-}" = wer ] && [ -z "${T:-}" ]; then
    read -r T NFE G < <(python scripts/select_wer_setting.py --run "$SWEEP_RUN" --step "${SWEEP_STEP:-0400000}")
    SUFFIX=_werSel
fi
if [ -z "${T:-}" ]; then
    read -r T NFE G < <(python - "$SWEEP_RUN" <<'PY'
import glob, json, sys
best = None
for f in glob.glob(f"results/fact_recon/sweep/{sys.argv[1]}_T*_N*_G*.json"):
    if f.endswith("_K8.json"):
        continue
    r = json.load(open(f))
    m = r["views"]["full"]["mel_l1_band"]["mean"]
    if best is None or m < best[0]:
        best = (m, r["temperature"], r["nfe"], r["cfg_scale"])
if best is None:
    sys.exit("no dev-sweep results: run scripts/submit_dev_sweep.sh, or set T, NFE and G")
print(best[1], best[2], best[3])
PY
)
fi
NFE=${NFE:-32}; G=${G:-1.0}
echo "vocoder: $VOC"
echo "sampling: temperature=$T nfe=$NFE cfg=$G (from ${SWEEP_RUN} dev sweep unless set)"
mkdir -p logs results/baseline_recon_ls results/baseline_recon_emilia
queued=$(squeue --me -h -o "%j" 2>/dev/null || true)
EXTRA_M="--wer-model large-v3 --utmos --sim"
# Speaker reference: another utterance of the same speaker (fair vs codecs
# without side information). Emilia-test keeps only speakers with two
# utterances (--partner-subset); tables then compare all rows on that subset.
SPK_REF=${SPK_REF:-other}

submit() {  # spec label
    local spec=$1 label=$2
    for SET in ls em; do
        local out name
        if [ $SET = ls ]; then out=results/baseline_recon_ls/$label.json; else out=results/baseline_recon_emilia/$label.json; fi
        name=fact-eb-$SET-$label
        if [ -e "$out" ]; then echo "  done    $name"; continue; fi
        if grep -qx "$name" <<<"$queued"; then echo "  queued  $name"; continue; fi
        local id
        if [ $SET = ls ]; then
            id=$(BASELINES="$spec:label=$label" EVAL_DIR="$LS" OUT=results/baseline_recon_ls EXTRA="$EXTRA_M" \
                 sbatch --parsable --account="$ACCOUNT" --job-name="$name" --time=16:00:00 slurm/eval_baselines.sbatch)
        else
            id=$(BASELINES="$spec:label=$label" EVAL_MANIFEST="$EM" OUT=results/baseline_recon_emilia \
                 EXTRA="$EXTRA_M $([ "$SPK_REF" = other ] && [ "${2#vocoder}" = "$2" ] && echo --partner-subset)" \
                 sbatch --parsable --account="$ACCOUNT" --job-name="$name" --time=16:00:00 slurm/eval_baselines.sbatch)
        fi
        echo "  SUBMIT  $name -> $id"
    done
}

echo "== vocoder copy-synthesis (ceiling)"
submit "fact_vocoder:vocoder=$VOC" "vocoder_copysyn"
for R in $RUNS; do
    CK=$D/runs/$R/step_$STEP.pt
    [ -f "$CK" ] || { echo "$R: no step_$STEP.pt, skipped"; continue; }
    native=$(python - "$CK" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=True)["config"]["bottleneck"]
print("discrete_only" if c.get("residual_dropout", 0) > 0 else "tokens_prior")
PY
)
    echo "== $R (FaCT + vocoder; views full, $native)"
    for V in full $native; do
        submit "fact:ckpt=$CK:vocoder=$VOC:view=$V:nfe=$NFE:temperature=$T:cfg_scale=$G:spk_ref=$SPK_REF" \
            "fact_${R}${STEP_TAG}_${V}_spk${SPK_REF}${SUFFIX}"
    done
done
if [ -n "${WAVE:-}" ]; then
    WR=${WAVE_RUN:?set WAVE_RUN=<run the decoder was trained on>}
    echo "== $WR (FaCT + stage-D waveform decoder)"
    for V in full discrete_only; do
        submit "fact_wave:decoder=$WAVE:view=$V:spk_ref=$SPK_REF" "factwave_${WR}_${V}_spk${SPK_REF}"
    done
fi
