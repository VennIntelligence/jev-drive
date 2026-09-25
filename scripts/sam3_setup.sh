#!/usr/bin/env bash
# SAM 3.1 environment + weights for the fusion diagnostics (todos/2026-09-25-fusion-diagnostics.md, Q4).
#   - code: facebookresearch/sam3 cloned to $DATA_DIR/third_party/sam3 (GitHub via network_turbo)
#   - env:  $DATA_DIR/envs/sam3 (Python 3.12, torch 2.10 cu128, sam3 editable + the few extras we use)
#   - weights: ModelScope copies of facebook/sam3.1 (and facebook/sam3, the image-detector fallback) into
#     $DATA_DIR/models/{sam3.1,sam3}. HF's copy is manual-gated for our account; the SAM License was accepted
#     by the user on 2026-09-25.
# Idempotent: re-running skips what is already there.
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
SRC=$DATA_DIR/third_party/sam3 ENV=$DATA_DIR/envs/sam3

if [[ ! -d $SRC/.git ]]; then
  (source /etc/network_turbo >/dev/null; git clone --depth 1 https://github.com/facebookresearch/sam3 "$SRC")
fi
echo "sam3 code at $(git -C "$SRC" rev-parse HEAD)"

fetch() {  # ModelScope is domestic: always direct
  (unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
   modelscope download "$1" --local-dir "$DATA_DIR/models/$2" --max-workers 4)
}
fetch facebook/sam3.1 sam3.1 &
dl=$!

if [[ ! -x $ENV/bin/python ]]; then
  uv venv --python 3.12 "$ENV"
fi
export VIRTUAL_ENV=$ENV
uv pip install --python "$ENV/bin/python" torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128 \
  || (source /etc/network_turbo >/dev/null; uv pip install --python "$ENV/bin/python" torch==2.10.0 torchvision \
      --index-url https://download.pytorch.org/whl/cu128)
uv pip install --python "$ENV/bin/python" -e "$SRC" einops pandas pyarrow opencv-python-headless pycocotools \
  scipy tqdm tensorboard
wait $dl
fetch facebook/sam3 sam3 || echo "facebook/sam3 fallback not fetched (only needed if 3.1 has no image path)"
"$ENV/bin/python" -c "import torch, sam3; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available())"
ls -la "$DATA_DIR/models/sam3.1" "$DATA_DIR/models/sam3" || true
