#!/usr/bin/env bash
# Download the pretrained models this project uses into the HF cache ($HF_HOME),
# plus a few large baselines from ModelScope into $DATA_DIR/models/<name> (MS_MODELS).
# Run on the GPU box, ideally inside tmux. Safe to re-run: finished files are skipped.
# Usage: scripts/download_models.sh [repo ...]      (default: every repo in MODELS and MS_MODELS)
#        scripts/download_models.sh ms:<repo> ...  (one ModelScope repo)
# Source: hf-mirror.com by default (domestic, fast). JEV_HF_VIA=turbo uses huggingface.co via network_turbo.
# JEV_HF_WORKERS (default 2) caps concurrent files so parallel dataset downloads are not starved.
# Gated repos (e.g. DINOv3) need a token: put HUGGING_FACE=hf_... in $DATA_DIR/jev-drive/.env
# (git-ignored, chmod 600). Never echo it; a 403 on a gated repo usually means the access request
# is still pending, not that the token is wrong.
set -euo pipefail

MODELS=(
  # Our feature backbones
  Qwen/Qwen3-VL-4B-Instruct
  Qwen/Qwen3-VL-8B-Instruct
  Qwen/Qwen3-VL-32B-Instruct
  facebook/dinov2-base
  facebook/dinov3-vitb16-pretrain-lvd1689m   # gated: needs an approved access request
  facebook/dinov3-vitl16-pretrain-lvd1689m   # gated: needs an approved access request
  # Baselines benchmarked in docs/baselines.md
  nvidia/diffusiongemma-26B-A4B-it-NVFP4   # openjev
  Zewei-Zhou/AutoVLA                       # AutoVLA NAVSIM checkpoint (UCLA academic licence: academic use only)
)
# From ModelScope: hf-mirror redirects these to Xet's US CDN (~0.3 MB/s here), ModelScope is domestic.
MS_MODELS=(
  Qwen/Qwen-Drive-1.0-4B          # -> $DATA_DIR/models/Qwen-Drive-1.0-4B
  Qwen/Qwen2.5-VL-3B-Instruct     # AutoVLA's base model -> $DATA_DIR/models/Qwen2.5-VL-3B-Instruct
)
if (( $# )); then
  MODELS=() MS_MODELS=()
  for r in "$@"; do [[ $r == ms:* ]] && MS_MODELS+=("${r#ms:}") || MODELS+=("$r"); done
fi

: "${DATA_DIR:?DATA_DIR is not set, see docs/storage.md}"
if (( ${#MS_MODELS[@]} )); then
  command -v modelscope >/dev/null || uv tool install modelscope-hub   # the CLI lives in modelscope-hub
  for repo in "${MS_MODELS[@]}"; do
    echo "==> modelscope $repo"
    (unset http_proxy https_proxy; modelscope download "$repo" --local-dir "$DATA_DIR/models/${repo#*/}" \
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

failed=()
for repo in "${MODELS[@]}"; do
  echo "==> $repo"
  # hf-mirror returns 403 for some junk files (e.g. .DS_Store in Qwen-Drive); skip them.
  if ! hf download "$repo" --max-workers "${JEV_HF_WORKERS:-2}" --exclude '.DS_Store' --exclude '.ms_upload_cache*'; then
    failed+=("$repo")   # one gated repo must not abort the rest
  fi
done
if (( ${#failed[@]} )); then
  echo "!! not downloaded: ${failed[*]}"
  echo "   For a gated repo this is usually a pending access request: open its page on huggingface.co"
  echo "   with the account that owns the token and check that access is granted (not just requested)."
fi

echo "==> done, cache size:"
du -sh "$HF_HOME/hub"
