#!/usr/bin/env bash
# Build the AutoVLA venv on the box. Code: $DATA_DIR/third_party/autovla (ucla-mobility/AutoVLA @ ba34eed),
# venv: $DATA_DIR/envs/autovla. Licence: UCLA Academic Software License, academic / non-profit use only.
#
# Changed versus their pins (requirements.txt, environment.yml), all forced by Blackwell (sm_120):
#   python 3.9 -> 3.12, torch 2.4.0 -> 2.8.0+cu128, torchvision 0.19.0 -> 0.23.0, triton 3.0.0 -> the one torch pins.
#   Everything else stays at their pin (transformers 4.49.0, tokenizers 0.21.1, accelerate 1.5.2,
#   qwen-vl-utils 0.0.10, pytorch-lightning 2.2.1, numpy 1.x). nuplan-devkit / navsim and the geo stack are
#   NOT installed: inference never touches them (see autovla.py, which stubs models.utils.score).
set -euo pipefail
: "${DATA_DIR:?}"
src=$DATA_DIR/third_party/autovla env=$DATA_DIR/envs/autovla
commit=ba34eed74ce6729e7986592d0e66cbaca397b4fa
if [[ ! -d $src ]]; then
  set +u; source /etc/network_turbo >/dev/null; set -u
  git clone --filter=blob:none https://github.com/ucla-mobility/AutoVLA "$src"
  unset http_proxy https_proxy
fi
git -C "$src" checkout -q $commit
[[ -x $env/bin/python ]] || (cd /tmp && uv venv -p 3.12 "$env")
export VIRTUAL_ENV=$env UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple UV_CONCURRENT_DOWNLOADS=3
uv pip install torch==2.8.0 torchvision==0.23.0
uv pip install transformers==4.49.0 tokenizers==0.21.1 accelerate==1.5.2 qwen-vl-utils==0.0.10 \
  pytorch-lightning==2.2.1 "numpy<2" pyyaml pillow av "setuptools<81"   # PL 2.2.1 still imports pkg_resources
"$env/bin/python" - <<'PY'
import torch, transformers, pytorch_lightning
print(torch.__version__, transformers.__version__, pytorch_lightning.__version__, torch.cuda.get_device_capability())
PY
