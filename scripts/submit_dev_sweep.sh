#!/bin/bash
# Inference-setting sweep on the FULL Emilia dev set (2,000 utterances; the
# model-selection set - never the test sets). One GPU job per setting;
# skips settings whose JSON exists or whose job is queued.
#
#   bash scripts/submit_dev_sweep.sh                    # proposed run
#   RUNS="main_c8_rd05 fact_main" bash scripts/submit_dev_sweep.sh
#
# Grid: temperature x NFE x CFG scale, plus one 8-sample job (mean-of-8
# diagnostic) at the trained setting. Views: full + the run's native
# tokens-only view. Output: results/fact_recon/sweep/<run>_T<t>_N<nfe>_G<cfg>[_K8].json
# Selection rule (fixed before results): lowest dev full-view band mel-L1;
# report the tokens-only view at the same setting. Table:
#   python scripts/sweep_table.py --out results/tables/dev_sweep.md
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
D=${D:-/leonardo_scratch/large/userexternal/amehrish/fact}
ACCOUNT=${ACCOUNT:-AIFAC_S07_041}
RUNS=${RUNS:-main_c8_rd05}
DEV=${DEV:-$D/manifests/dev.tsv}
TEMPS=${TEMPS:-"1.0 0.8 0.667 0.5"}
NFES=${NFES:-"32 64"}
CFGS=${CFGS:-"1.0 1.5"}
[ -f "$DEV" ] || { echo "missing $DEV"; exit 1; }
mkdir -p logs results/fact_recon/sweep
queued=$(squeue --me -h -o "%j" 2>/dev/null || true)

submit() {  # name out extra-args
    local name=$1 out=$2 extra=$3
    if [ -e "$out" ]; then echo "  done    $name"; return; fi
    if grep -qx "$name" <<<"$queued"; then echo "  queued  $name"; return; fi
    local id
    id=$(CKPT="$CK" EVAL_MANIFEST="$DEV" OUT="$out" EXTRA="$extra" \
         sbatch --parsable --account="$ACCOUNT" --job-name="$name" --time=12:00:00 slurm/eval_fact.sbatch)
    echo "  SUBMIT  $name -> $id"
}

for R in $RUNS; do
    CK=$D/runs/$R/step_0400000.pt
    [ -f "$CK" ] || { echo "$R: no step_0400000.pt"; continue; }
    native=$(python - "$CK" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=True)["config"]["bottleneck"]
print("discrete_only" if c.get("residual_dropout", 0) > 0 else "tokens_prior")
PY
)
    echo "== $R (views full,$native)"
    for T in $TEMPS; do for N in $NFES; do for G in $CFGS; do
        tag=${R}_T${T}_N${N}_G${G}
        submit "fact-sw-$tag" "results/fact_recon/sweep/$tag.json" \
            "--views full,$native --temperature $T --nfe $N --cfg-scale $G"
    done; done; done
    tag=${R}_T1.0_N32_G1.0_K8
    submit "fact-sw-$tag" "results/fact_recon/sweep/$tag.json" \
        "--views full,$native --temperature 1.0 --nfe 32 --cfg-scale 1.0 --n-samples 8"
done
