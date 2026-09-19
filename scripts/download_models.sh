#!/usr/bin/env bash
# Download the pretrained models this project uses into the HF cache ($HF_HOME),
# plus a few large baselines from ModelScope into $DATA_DIR/models/<name> (MS_MODELS).
# Run on the GPU box, ideally inside tmux. Safe to re-run: finished files are skipped.
# Usage: scripts/download_models.sh [repo ...]      (default: every repo in MODELS and MS_MODELS)
#        scripts/download_models.sh ms:<repo> ...  (one ModelScope repo)
# Source: hf-mirror.com by default (domestic, fast). JEV_HF_VIA=turbo uses huggingface.co via network_turbo.
# JEV_HF_WORKERS (default 2) caps concurrent files so parallel dataset downloads are not starved.
set -euo pipefail

MODELS=(
  # Our feature backbones
  Qwen/Qwen3-VL-4B-Instruct
  Qwen/Qwen3-VL-8B-Instruct
  Qwen/Qwen3-VL-32B-Instruct
  facebook/dinov2-base
  # Baselines benchmarked in docs/baselines.md
  nvidia/diffusiongemma-26B-A4B-it-NVFP4   # openjev
  # Not downloaded on purpose: Zewei-Zhou/AutoVLA (academic/nonprofit-only license, see docs/baselines.md)
)
# From ModelScope: hf-mirror redirects these to Xet's US CDN (~0.3 MB/s here), ModelScope is domestic.
MS_MODELS=(
  Qwen/Qwen-Drive-1.0-4B   # -> $DATA_DIR/models/Qwen-Drive-1.0-4B
)
if (( $# )); then
  MODELS=() MS_MODELS=()
  for r in "$@"; do [[ $r == ms:* ]] && MS_MODELS+=("${r#ms:}") || MODELS+=("$r"); done
fi

: "${DATA_DIR:?DATA_DIR is not set, see docs/storage.md}"
if (( ${#MS_MODELS[@]} )); then
  command -v modelscope >/dev/null || uv tool install modelscope
  for repo in "${MS_MODELS[@]}"; do
    echo "==> modelscope $repo"
    (unset http_proxy https_proxy; modelscope download --model "$repo" --local_dir "$DATA_DIR/models/${repo#*/}" \
      --exclude '.DS_Store' --max-workers "${JEV_HF_WORKERS:-2}")
  done
fi

: "${HF_HOME:?HF_HOME is not set, see docs/storage.md}"

# Install the CLI before turning on any proxy: turbo slows down pip and uv.
if ! command -v hf >/dev/null; then
  uv tool install "huggingface_hub[cli]"
fi

if [[ "${JEV_HF_VIA:-mirror}" == turbo ]]; then
  if [[ -f /etc/network_turbo && -z "${https_proxy:-}" ]]; then
    set +u; source /etc/network_turbo >/dev/null; set -u
  fi
else
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
fi

# Xet storage returns 401 through the AutoDL proxies; plain HTTP downloads work.
export HF_HUB_DISABLE_XET=1

for repo in "${MODELS[@]}"; do
  echo "==> $repo"
  # hf-mirror returns 403 for some junk files (e.g. .DS_Store in Qwen-Drive); skip them.
  hf download "$repo" --max-workers "${JEV_HF_WORKERS:-2}" --exclude '.DS_Store' --exclude '.ms_upload_cache*'
done

echo "==> done, cache size:"
du -sh "$HF_HOME/hub"
