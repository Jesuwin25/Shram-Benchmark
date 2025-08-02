#!/usr/bin/env bash
set -euo pipefail

# —— Configuration ——  
MODELS=(gemma3:1b qwen2.5:1.5b llama3:1b)  
TEMPS=(0.0 0.7 1.0)  
PROMPT="Write a brief story about Shram’s mission and its impact on work using the provided context."  
NUM_PREDICT=256  
FACT_DATASET="shram_ai_dataset.json"  
FACT_PREDICT=64  
WEIGHT_FACT=0.6  
WEIGHT_EFF=0.4  

run_bench() {
  local BASE_URL=$1
  local SUFFIX=$2
  local OUTFILE="extended_results_$(hostname)_${SUFFIX}.json"
  echo "▶ Running benchmark on $SUFFIX @ $BASE_URL ..."
  python3 benchmark_extended.py \
    --base-url "$BASE_URL" \
    --models "${MODELS[@]}" \
    --temperatures "${TEMPS[@]}" \
    --prompt "$PROMPT" \
    --num-predict "$NUM_PREDICT" \
    --factuality-dataset "$FACT_DATASET" \
    --factuality-predict "$FACT_PREDICT" \
    --weight-factuality "$WEIGHT_FACT" \
    --weight-efficiency "$WEIGHT_EFF" \
    --output "$OUTFILE"
  echo "✅ Done — results in $OUTFILE"
}

# ---- Run on your GPU-enabled Ollama server ----
run_bench http://localhost:11434 GPU

# ---- (Optional) CPU-only run — start Ollama with `ollama serve --no-gpu --port 11435` ----
# run_bench http://localhost:11435 CPU
