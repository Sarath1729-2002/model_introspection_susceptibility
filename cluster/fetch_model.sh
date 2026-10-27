#!/bin/bash
# Run ON THE LOGIN NODE (compute nodes have no internet; HF_HUB_OFFLINE=1 there).  Downloads the second model of the paper matrix into the shared HF cache.
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
HF_HUB_OFFLINE=0 source cluster/config.sh
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
source "$VENV/bin/activate"
python - <<'P'
import os
from huggingface_hub import snapshot_download
mid = os.environ.get("R1_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
p = snapshot_download(mid, allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*", "*.txt"])
print("cached at", p)
from transformers import AutoConfig, AutoTokenizer
c = AutoConfig.from_pretrained(mid); t = AutoTokenizer.from_pretrained(mid)
print("layers", c.num_hidden_layers, "hidden", c.hidden_size, "chat template:", bool(t.chat_template))
print(t.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True))
P
