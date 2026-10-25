#!/bin/bash
# ONE-TIME, on the LOGIN node (needs internet), from the repo root:   bash cluster/setup.sh
# Creates the venv, caches the model, downloads GSM8K / MATH / ARC, generates the synthetic sets, runs the GROUND-TRUTH AUDIT and writes problems.jsonl.
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source cluster/config.sh
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0
module purge 2>/dev/null || true
module load StdEnv/2023 gcc arrow python/3.12 2>/dev/null || echo "[warn] module load failed - try: module spider arrow python"
mkdir -p "$ROOT/outputs" "$HF_HOME" "$(dirname "$VENV")"
[[ -d "$VENV" ]] || python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install --no-index "torch>=2.9.0" 2>/dev/null || pip install "torch>=2.9.0"
pip install -e .
mkdir -p "$RUN_DIR"
python - <<PY
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["MODEL_ID"], allow_patterns=["*.json", "*.safetensors", "*.txt"])
print("model cached")
PY
python -m susc.cli data --run-dir "$RUN_DIR" --sizes "$SIZES"
echo; echo "=== ground-truth audit ==="; python -c "import json;[print(a['dataset'], 'tier', a['tier'], 'kept', a['n_kept'], '/', a['n_in'], 'ADMITTED' if a['admitted'] else 'REJECTED', a['fails']) for a in json.load(open('$RUN_DIR/ground_truth_audit.json'))]"
wc -l "$RUN_DIR/problems.jsonl"
echo "next:  MODE=gate bash cluster/submit.sh   (optional ~1h dry run)   or   MODE=full bash cluster/submit.sh"
