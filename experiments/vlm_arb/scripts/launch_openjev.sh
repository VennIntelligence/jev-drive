#!/usr/bin/env bash
# Launch vLLM server with DiffusionGemma-26B NVFP4 and OpenJev System One API.
set -euo pipefail
: "${DATA_DIR:=/root/autodl-tmp/ujs}"

env=$DATA_DIR/envs/openjev
export PATH=$env/bin:$PATH
export HF_HUB_OFFLINE=1
export VLLM_CACHE_ROOT=$DATA_DIR/cache/vllm
MODEL=${OPENJEV_MODEL:-nvidia/diffusiongemma-26B-A4B-it-NVFP4}
export OPENJEV_TOKENIZER=$MODEL
export OPENJEV_CANVAS=${OPENJEV_CANVAS:-64}

log_dir=$DATA_DIR/runs/vlm_arb/logs
mkdir -p "$log_dir"
log=$log_dir/openjev-$(date +%Y%m%d-%H%M%S).log

echo "Starting vLLM ($MODEL) on GPU utilization ${OPENJEV_GPU_UTIL:-0.25}..."
vllm serve "$MODEL" --served-model-name dgemma --host 127.0.0.1 --port 8000 \
  --diffusion-config "{\"canvas_length\": ${OPENJEV_CANVAS}}" --max-logprobs 32 \
  --limit-mm-per-prompt '{"image": 8, "video": 0}' \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --override-generation-config '{"max_new_tokens": null}' --enable-prefix-caching --async-scheduling \
  --attention-backend TRITON_ATTN --max-num-seqs 64 --max-model-len 65536 \
  --gpu-memory-utilization "${OPENJEV_GPU_UTIL:-0.25}" >"$log" 2>&1 &
vllm_pid=$!

trap 'kill -TERM $(jobs -p) 2>/dev/null; wait' EXIT

echo "Waiting for vLLM on port 8000 (log: $log)..."
until curl -sf http://127.0.0.1:8000/health >/dev/null; do
  kill -0 $vllm_pid 2>/dev/null || { tail -50 "$log"; echo "vLLM exited during startup" >&2; exit 1; }
  sleep 4
done
echo "vLLM is healthy."

echo "Warming up openjev..."
python -m openjev.warmup

echo "Starting openjev System One server on port 8080..."
python -m openjev >>"$log" 2>&1 &
until curl -sf http://127.0.0.1:8080/health >/dev/null; do sleep 1; done
echo "OpenJev is healthy on http://127.0.0.1:8080."

wait
