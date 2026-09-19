#!/usr/bin/env bash
# Download the pretrained models this project uses into the HF cache ($HF_HOME).
# Run on the GPU box, ideally inside tmux. Safe to re-run: finished files are skipped.
set -euo pipefail

MODELS=(
  Qwen/Qwen3-VL-4B-Instruct
)

: "${HF_HOME:?HF_HOME is not set, see docs/storage.md}"

# Install the CLI before turning on the proxy: turbo slows down pip and uv.
if ! command -v hf >/dev/null; then
  uv tool install "huggingface_hub[cli]"
fi

if [[ -f /etc/network_turbo && -z "${https_proxy:-}" ]]; then
  set +u
  source /etc/network_turbo >/dev/null
  set -u
fi

for repo in "${MODELS[@]}"; do
  echo "==> $repo"
  hf download "$repo"
done

echo "==> done, cache size:"
du -sh "$HF_HOME/hub"
