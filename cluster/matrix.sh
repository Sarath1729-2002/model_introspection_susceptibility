#!/bin/bash
# THE PAPER MATRIX.  Cells = (model x dataset).  One command per cell chain; all resumable (finished chunk files are skipped, resubmitting is safe).
#   bash cluster/fetch_model.sh                       # login node, once: downloads DeepSeek-R1-Distill-Qwen-7B into the HF cache (compute nodes are offline)
#   CELLS="qwen-gsm8k qwen-math r1-gsm8k r1-math" bash cluster/matrix.sh     # everything; independent chains are submitted at once
#   SMOKE=1 CELLS="r1-gsm8k qwen-math" bash cluster/matrix.sh   # ~1-2h dry run: 40 problems/dataset, tiny budgets, writes to *_psmoke dirs (qwen-gsm8k has no smoke: it needs the real inject outputs)
#   bash cluster/matrix.sh paper                      # assemble PAPER_REPORT.md from whatever cells have finished (afterany of nothing; run when jobs are done)
# Cell definitions
#   qwen-gsm8k  existing run (mega/ + robust/): only the extras - stoch, natural, cell-analyze
#   qwen-math   same run dir, MATH problems already generated: steps -> inject -> robust-prep -> robust-steer || stoch || natural -> cell-analyze   (mega_math/)
#   r1-gsm8k, r1-math   DeepSeek-R1-Distill-Qwen-7B: shared gen+states (no constrained prompt) then the same chain per dataset (run dir r1d7b-psuite)
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source cluster/config.sh
COMMON=(--parsable --account="$SBATCH_ACCOUNT" --export=ALL --mem=0)
[[ -n "$SLACK_EMAIL" ]] && COMMON+=(--mail-user="$SLACK_EMAIL" --mail-type=FAIL,TIME_LIMIT_80)
NODE=(--gpus-per-node=h100:4 --cpus-per-task=24)
sub() { local stage="$1" t="$2" dep="$3"; shift 3
  local d=(); [[ -n "$dep" ]] && d=(--dependency="${DEPTYPE:-afterok}":"$dep")
  STAGE="$stage" sbatch "${COMMON[@]}" --job-name="psuite-$stage-${CELL_TAG:-x}" --time="$t" ${d[@]+"${d[@]}"} "$@" cluster/run.sbatch; }
hh() { python3 -c "h=float('$1')+0.75;print('%02d:%02d:00'%(int(h),int((h-int(h))*60)))"; }
SMOKE="${SMOKE:-}"
R1_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
QW_RUN="$ROOT/outputs/qwen7b-psuite"; R1_RUN="$ROOT/outputs/r1d7b-psuite"
sfx=""; [[ -n "$SMOKE" ]] && sfx="_psmoke"
SPEC="$ROOT/outputs/paper_cells${sfx}.json"

cell_env() {            # cell_env <cell>   sets the environment of one cell
  case "$1" in
    qwen-gsm8k) export MODEL_ID="Qwen/Qwen2.5-7B-Instruct" RUN_NAME=qwen7b-psuite RUN_DIR="$QW_RUN" CELL_DS=gsm8k MEGA_DIR="$QW_RUN/mega" ROBUST_DIR=robust ;;
    qwen-math)  export MODEL_ID="Qwen/Qwen2.5-7B-Instruct" RUN_NAME=qwen7b-psuite RUN_DIR="$QW_RUN" CELL_DS=math  MEGA_DIR="$QW_RUN/mega_math" ROBUST_DIR=robust ;;
    r1-gsm8k)   export MODEL_ID="$R1_ID" RUN_NAME=r1d7b-psuite RUN_DIR="$R1_RUN" CELL_DS=gsm8k MEGA_DIR="$R1_RUN/mega" ROBUST_DIR=robust ;;
    r1-math)    export MODEL_ID="$R1_ID" RUN_NAME=r1d7b-psuite RUN_DIR="$R1_RUN" CELL_DS=math  MEGA_DIR="$R1_RUN/mega_math" ROBUST_DIR=robust ;;
    *) echo "unknown cell $1"; exit 2 ;;
  esac
  export BASE_RUN_NAME=qwen7b-psuite CELL_TAG="${1}${sfx}"
  if [[ "$1" == r1-* ]]; then       # long chains of thought: big token budgets, small batches, fewer/cheaper edits, no constrained prompt, no deletion edit
    export CELL_NO_CON=1 CELL_NO_DEL=1 MAX_SEQS=32 GEN_MAX_NEW=6144 STEPS_MAX_NEW=6144 ROBUST_MAX_NEW=6144 MEGA_MAX_STEPS=6 MEGA_CHUNK=4 N_SPREAD=2
    export CELL_LIMIT="${R1_LIMIT:-500}" ROBUST_N="${R1_ROBUST_N:-1500}" STOCH_N="${R1_STOCH_N:-500}" STOCH_NSAMP=6
    export GEN_BUDGET_H="${R1_GEN_H:-3}" MEGA_STEPS_H="${R1_STEPS_H:-3.5}" INJECT_H=1.5 ROBUST_STEER_H="${R1_STEER_H:-3}" STOCH_H=2 NATURAL_H=2
  else
    unset CELL_NO_CON CELL_NO_DEL CELL_LIMIT
    export STEPS_MAX_NEW=512 ROBUST_MAX_NEW="${ROBUST_MAX_NEW:-384}" STOCH_H=1.5 NATURAL_H=1.5
    [[ "$1" == qwen-math ]] && export MEGA_STEPS_H="${QM_STEPS_H:-3}" MEGA_MAX_STEPS=6 ROBUST_N="${QM_ROBUST_N:-2000}" CELL_LIMIT="${QM_LIMIT:-800}" STEPS_MAX_NEW=1024 ROBUST_MAX_NEW=1024 MAX_SEQS=64
  fi
  if [[ -n "$SMOKE" ]]; then
    export CELL_LIMIT=40 ROBUST_N=64 STOCH_N=48 STOCH_NSAMP=4 MEGA_STEPS_H=0.15 INJECT_H=0.1 ROBUST_STEER_H=0.15 STOCH_H=0.15 NATURAL_H=0.15 GEN_BUDGET_H=0.2 MEGA_MAX_STEPS=3 MEGA_CHUNK=2
    export ROBUST_DIR=robust_smoke
    [[ "$1" == r1-* ]] && export RUN_DIR="${R1_RUN}${sfx}" RUN_NAME="r1d7b-psuite${sfx}"
    export MEGA_DIR="${RUN_DIR}/mega$([[ "$CELL_DS" == math ]] && echo _math)${sfx}"
  fi
}

chain() {               # chain <cell> <dependency-for-first-stage or ''>   -> echoes the cell-analyze job id
  local c="$1" dep="${2:-}" j1 j2 j3 j4 j5 ja
  ( cell_env "$c"
    T() { if [[ -n "$SMOKE" ]]; then echo 01:00:00; else hh "$1"; fi; }
    if [[ "$c" == qwen-gsm8k ]]; then
      j3=$(sub stoch-run "$(T "$STOCH_H")" "$dep" "${NODE[@]}"); j4=$(sub natural-run "$(T "$NATURAL_H")" "$j3" "${NODE[@]}")
      ja=$(DEPTYPE=afterany sub cell-analyze 02:00:00 "$j4" "${NODE[@]}")
    else
      j1=$(sub mega-steps "$(T "$MEGA_STEPS_H")" "$dep" "${NODE[@]}")
      j2=$(sub inject "$(T "$INJECT_H")" "$j1" "${NODE[@]}")
      j3=$(sub robust-prep 00:45:00 "$j2" "${NODE[@]}")
      j4=$(DEPTYPE=afterany sub robust-steer "$(T "$ROBUST_STEER_H")" "$j3" "${NODE[@]}")
      j5=$(DEPTYPE=afterany sub stoch-run "$(T "$STOCH_H")" "$j4" "${NODE[@]}")
      j5=$(DEPTYPE=afterany sub natural-run "$(T "$NATURAL_H")" "$j5" "${NODE[@]}")
      j3=$(DEPTYPE=afterany sub robust-offline 03:00:00 "$j2" "${NODE[@]}")
      ja=$(DEPTYPE=afterany sub cell-analyze 02:00:00 "$j5:$j3" "${NODE[@]}")
    fi
    echo "$ja" )
}

write_spec() {          # the cell list consumed by the paper stage
  mkdir -p "$(dirname "$SPEC")"
  python3 - "$SPEC" "$sfx" <<'P'
import json, os, sys
spec, sfx = sys.argv[1], sys.argv[2]
root = os.environ["ROOT"] + "/outputs"
q, r = root + "/qwen7b-psuite", root + "/r1d7b-psuite" + sfx
cells = [("Qwen2.5-7B / GSM8K", "qwen7b", "gsm8k", q, q + "/mega" + sfx, "robust" + ("_smoke" if sfx else "")),
         ("Qwen2.5-7B / MATH", "qwen7b", "math", q, q + "/mega_math" + sfx, "robust" + ("_smoke" if sfx else "")),
         ("R1-Distill-7B / GSM8K", "r1d7b", "gsm8k", r, r + "/mega" + sfx, "robust" + ("_smoke" if sfx else "")),
         ("R1-Distill-7B / MATH", "r1d7b", "math", r, r + "/mega_math" + sfx, "robust" + ("_smoke" if sfx else ""))]
json.dump([{"name": n, "model": m, "ds": d, "run_dir": a, "mega_dir": b, "robust": rb} for n, m, d, a, b, rb in cells], open(spec, "w"), indent=1)
P
}

if [[ "${1:-}" == "paper" ]]; then
  write_spec
  ( cell_env qwen-gsm8k; export CELLS_JSON="$SPEC" PAPER_OUT="$ROOT/outputs/paper${sfx}"; ja=$(sub paper 01:30:00 "" "${NODE[@]}"); echo "paper: $ja -> $PAPER_OUT/PAPER_REPORT.md" )
  exit 0
fi
CELLS="${CELLS:-qwen-gsm8k qwen-math r1-gsm8k r1-math}"
write_spec
ids=()
r1_dep=""
if [[ " $CELLS " == *" r1-"* ]]; then       # shared by both R1 cells: plain greedy solutions + spread samples + prompt-end states (no constrained prompt)
  r1_dep=$( ( cell_env r1-gsm8k
              g=$(DATASETS=gsm8k,math sub gen "$([[ -n "$SMOKE" ]] && echo 01:00:00 || hh "$GEN_BUDGET_H")" "" "${NODE[@]}")
              s=$(DEPTYPE=afterany DATASETS=gsm8k,math sub states 02:00:00 "$g" "${NODE[@]}"); echo "$s" ) )
  echo "r1 gen+states: $r1_dep"
fi
for c in $CELLS; do
  case "$c" in r1-*) d="$r1_dep" ;; *) d="" ;; esac
  j=$(chain "$c" "$d"); ids+=("$j"); echo "cell $c -> final analyze job $j"
done
echo "all cells submitted (squeue -u \$USER). When the cell-analyze jobs are done:  bash cluster/matrix.sh paper   ->  $ROOT/outputs/paper${sfx}/PAPER_REPORT.md"
notify "matrix SUBMITTED" "cells: $CELLS ; analyze jobs: ${ids[*]}"
