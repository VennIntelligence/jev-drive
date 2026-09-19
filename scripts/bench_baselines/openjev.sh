#!/usr/bin/env bash
# Latency of openjev as released: start vLLM + the openjev API exactly as docker/entrypoint.sh does
# (flags copied from openjev @ 91d5005), run the warmup module, then openjev_client.py, then stop both.
# Run on the box inside tmux: scripts/tmux_run.sh bl-openjev scripts/bench_baselines/openjev.sh
set -euo pipefail
: "${DATA_DIR:?}"
env=$DATA_DIR/envs/openjev
export PATH=$env/bin:$PATH HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=$DATA_DIR/cache/vllm
MODEL=${OPENJEV_MODEL:-nvidia/diffusiongemma-26B-A4B-it-NVFP4}
export OPENJEV_TOKENIZER=$MODEL OPENJEV_CANVAS=${OPENJEV_CANVAS:-64}
log=$DATA_DIR/runs/bench_baselines/openjev/server-$(date +%Y%m%d-%H%M%S).log
mkdir -p "$(dirname "$log")"
vllm serve "$MODEL" --served-model-name dgemma --host 127.0.0.1 --port 8000 \
  --diffusion-config "{\"canvas_length\": ${OPENJEV_CANVAS}}" --max-logprobs 32 \
  --limit-mm-per-prompt '{"image": 8, "video": 0}' \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --override-generation-config '{"max_new_tokens": null}' --enable-prefix-caching --async-scheduling \
  --attention-backend TRITON_ATTN --max-num-seqs 64 --max-model-len 65536 \
  --gpu-memory-utilization "${OPENJEV_GPU_UTIL:-0.9}" >"$log" 2>&1 &
vllm_pid=$!
trap 'kill -TERM $(jobs -p) 2>/dev/null; wait' EXIT
echo "waiting for vLLM (log: $log)"
until curl -sf http://127.0.0.1:8000/health >/dev/null; do
  kill -0 $vllm_pid 2>/dev/null || { tail -50 "$log"; echo "vLLM exited during startup" >&2; exit 1; }
  sleep 5
done
python -m openjev.warmup
python -m openjev >>"$log" 2>&1 &
until curl -sf http://127.0.0.1:8080/health >/dev/null; do sleep 1; done
python "$(dirname "$0")/openjev_client.py" "$@"
