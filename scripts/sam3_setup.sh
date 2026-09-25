#!/usr/bin/env bash
# SAM 3.1 environment + weights for the fusion diagnostics (todos/2026-09-25-fusion-diagnostics.md, Q4).
#   - code: facebookresearch/sam3 cloned to $DATA_DIR/third_party/sam3 (GitHub via network_turbo)
#   - env:  $DATA_DIR/envs/sam3 (Python 3.12, torch 2.13 cu130, sam3 editable + the few extras we use)
#   - weights: the ModelScope copy of facebook/sam3.1 (sam3.1_multiplex.pt, sha256-checked) into $DATA_DIR/models/sam3.1. HF's copy is manual-gated for our account; the SAM License was accepted
#     by the user on 2026-09-25.
# Idempotent: re-running skips what is already there.
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
SRC=$DATA_DIR/third_party/sam3 ENV=$DATA_DIR/envs/sam3

if [[ ! -d $SRC/.git ]]; then
  (source /etc/network_turbo >/dev/null; git clone --depth 1 https://github.com/facebookresearch/sam3 "$SRC")
fi
echo "sam3 code at $(git -C "$SRC" rev-parse HEAD)"

[[ -f $DATA_DIR/models/sam3.1/sam3.1_multiplex.pt ]] || {
  # parallel range requests (jevdrive.hfdl) instead of the modelscope CLI: ~10 MB/s against ~1 MB/s
  mkdir -p "$DATA_DIR/models/sam3.1"
  (cd "$(dirname "$0")/.." && env -u http_proxy -u https_proxy PYTHONPATH=. "$DATA_DIR/envs/jevdrive/bin/python" -c "
import os; from pathlib import Path; from jevdrive.hfdl import download, ms_url
download(ms_url('facebook/sam3.1', 'sam3.1_multiplex.pt'), Path(os.environ['DATA_DIR']) / 'models/sam3.1/sam3.1_multiplex.pt',
         3502755717, '0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6', streams=24, headers={})")
}

if [[ ! -x $ENV/bin/python ]]; then
  uv venv --python 3.12 "$ENV"
fi
export VIRTUAL_ENV=$ENV
# torch 2.13 cu130 (sm_120 supported; SAM 3 asks for >= 2.7): the wheel is already in the box's uv cache from other
# envs, while a fresh cu128 2.10 download from download.pytorch.org ran at < 1 MB/s (2026-09-25)
uv pip install --python "$ENV/bin/python" torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "$ENV/bin/python" -e "$SRC" einops pandas pyarrow opencv-python-headless pycocotools \
  scipy tqdm tensorboard
"$ENV/bin/python" -c "import torch, sam3; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available())"
ls -la "$DATA_DIR/models/sam3.1"
