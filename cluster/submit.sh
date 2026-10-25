#!/bin/bash
# From the repo root on the login node AFTER  bash cluster/setup.sh
#   MODE=gate    bash cluster/submit.sh   # optional dry run: GATE_LIMIT problems per dataset, every stage, <~1.5h incl. queue -> ${RUN_DIR}-gate/analysis/REPORT.md
#   MODE=full    bash cluster/submit.sh   # the study: states -> sample -> trace -> analyze -> report, chained (afterok).  Resubmitting is safe: finished chunks are skipped.
#   MODE=resume  bash cluster/submit.sh   # continue after the sampling time budget cut the run short (sample -> trace -> analyze -> report)
#   MODE=part    bash cluster/submit.sh   # states + sample only, for DATASETS=... (use on a second node; then MODE=resume does trace -> analyze -> report for everything)
#   MODE=rescue  RESCUE_SHARD=2 bash cluster/submit.sh   # a shard crashed: finish ITS leftover chunks on all 4 GPUs, then trace -> analyze -> report
#   MODE=analyze bash cluster/submit.sh   # re-run analysis + report on what exists
#   MODE=report  bash cluster/submit.sh   # just rebuild REPORT.md from the analysis pickles
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source cluster/config.sh
[[ -f "$RUN_DIR/problems.jsonl" ]] || { echo "run bash cluster/setup.sh first"; exit 1; }
MODE="${MODE:-gate}"
COMMON=(--parsable --account="$SBATCH_ACCOUNT" --export=ALL --mem=0)
[[ -n "$SLACK_EMAIL" ]] && COMMON+=(--mail-user="$SLACK_EMAIL" --mail-type=FAIL,TIME_LIMIT_80)
sub() { local stage="$1" t="$2" dep="$3"; shift 3
  local d=(); [[ -n "$dep" ]] && d=(--dependency=afterok:"$dep")
  STAGE="$stage" sbatch "${COMMON[@]}" --job-name="susc-$stage" --time="$t" ${d[@]+"${d[@]}"} "$@" cluster/run.sbatch; }
NODE=(--gpus-per-node=h100:4 --cpus-per-task=24)   # TamIA allocates whole nodes only
SAMPLE_T="$(python3 -c "import math;h=float('${SAMPLE_BUDGET_H}')+0.75;print('%02d:%02d:00'%(int(h),int((h-int(h))*60)))")"
case "$MODE" in
gate)
  export GATE=1
  st=$(sub states 01:00:00 "" "${NODE[@]}"); sa=$(sub sample 02:00:00 "$st" "${NODE[@]}"); tr=$(sub trace 01:00:00 "$sa" "${NODE[@]}")
  an=$(sub analyze 01:30:00 "$tr" "${NODE[@]}"); rp=$(sub report 00:20:00 "$an" "${NODE[@]}")
  echo "gate: states=$st sample=$sa trace=$tr analyze=$an report=$rp -> ${RUN_DIR}-gate/analysis/REPORT.md"; exit 0 ;;
full)
  st=$(sub states 03:00:00 "" "${NODE[@]}"); sa=$(sub sample "$SAMPLE_T" "$st" "${NODE[@]}"); tr=$(sub trace 02:00:00 "$sa" "${NODE[@]}")
  an=$(sub analyze 04:00:00 "$tr" "${NODE[@]}"); rp=$(sub report 00:30:00 "$an" "${NODE[@]}") ;;
part)
  st=$(sub states 03:00:00 "" "${NODE[@]}"); sa=$(sub sample "$SAMPLE_T" "$st" "${NODE[@]}"); tr="-"; an="-"; rp="-" ;;
resume)
  sa=$(sub sample "$SAMPLE_T" "" "${NODE[@]}"); tr=$(sub trace 02:00:00 "$sa" "${NODE[@]}")
  an=$(sub analyze 04:00:00 "$tr" "${NODE[@]}"); rp=$(sub report 00:30:00 "$an" "${NODE[@]}"); st="-" ;;
rescue)
  [[ -n "${RESCUE_SHARD:-}" ]] || { echo "set RESCUE_SHARD=<crashed shard number>"; exit 1; }
  sa=$(sub sample "$SAMPLE_T" "" "${NODE[@]}"); tr=$(sub trace 02:00:00 "$sa" "${NODE[@]}")
  an=$(sub analyze 04:00:00 "$tr" "${NODE[@]}"); rp=$(sub report 00:30:00 "$an" "${NODE[@]}"); st="-" ;;
analyze)
  an=$(sub analyze 04:00:00 "" "${NODE[@]}"); rp=$(sub report 00:30:00 "$an" "${NODE[@]}"); st="-"; sa="-"; tr="-" ;;
report)
  rp=$(sub report 00:30:00 "" "${NODE[@]}"); st="-"; sa="-"; tr="-"; an="-" ;;
*) echo "unknown MODE"; exit 2 ;;
esac
echo "$MODE: states=$st sample=$sa trace=$tr analyze=$an report=$rp   -> $RUN_DIR/analysis/REPORT.md   (squeue -u \$USER)"
notify "$MODE SUBMITTED" "states=$st sample=$sa trace=$tr analyze=$an report=$rp"
