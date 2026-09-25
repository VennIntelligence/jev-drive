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
# torch 2.13 cu130 (sm_120 supported; SAM 3 asks for >= 2.7). The torch / torchvision wheels are already in the box's
# uv cache from other envs; their CUDA runtime deps come from the domestic Aliyun PyPI mirror. A plain install from
# download.pytorch.org fetched every nvidia-* wheel from that CDN at < 1 MB/s (2026-09-25).
PT=https://download.pytorch.org/whl/cu130 MIRROR=https://mirrors.aliyun.com/pypi/simple
uv pip install --python "$ENV/bin/python" --no-deps torch==2.13.0+cu130 torchvision==0.28.0+cu130 --index-url $PT
deps=$("$ENV/bin/python" - <<'PY'
import importlib.metadata as m
print(" ".join(r.split(";")[0].replace(" ", "") for r in m.requires("torch") if "extra ==" not in r))
PY
)
uv pip install --python "$ENV/bin/python" $deps pillow --index-url $MIRROR
uv pip install --python "$ENV/bin/python" --index-url $MIRROR -e "$SRC" einops pandas pyarrow opencv-python-headless pycocotools \
  scipy tqdm tensorboard
"$ENV/bin/python" -c "import torch, sam3; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available())"
ls -la "$DATA_DIR/models/sam3.1"
