#!/usr/bin/env bash
# Build the openjev venv on the box without Docker, following openjev's docker/Dockerfile:
# vLLM fork razorback16/vllm @ 9bbf741 (PR vllm-project/vllm#57250 + fixes) with the upstream precompiled kernels
# of its merge-base 2c88fb1 (torch 2.13 cu130, sm_120 included), then openjev itself.
# Code: $DATA_DIR/third_party/{openjev,vllm-openjev}; venv: $DATA_DIR/envs/openjev.
set -euo pipefail
: "${DATA_DIR:?}"
tp=$DATA_DIR/third_party env=$DATA_DIR/envs/openjev
openjev_commit=91d5005effcf8cc0ecccaa9538ceabbb130fef59
vllm_commit=9bbf7418e85020dc76da9f60cdfe6c4e912ec048
vllm_base=2c88fb131c7ae0be01907cd8c276911db5e7aad4
set +u; source /etc/network_turbo >/dev/null; set -u   # GitHub only
[[ -d $tp/openjev ]] || git clone https://github.com/razorback16/openjev "$tp/openjev"
git -C "$tp/openjev" checkout -q $openjev_commit
[[ -d $tp/vllm-openjev ]] || git clone --filter=blob:none https://github.com/razorback16/vllm "$tp/vllm-openjev"
git -C "$tp/vllm-openjev" checkout -q $vllm_commit
unset http_proxy https_proxy
[[ -x $env/bin/python ]] || (cd /tmp && uv venv -p 3.12 "$env")
export VIRTUAL_ENV=$env UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple UV_CONCURRENT_DOWNLOADS=3
cd "$tp/vllm-openjev"
# flashinfer-python/-cubin come from PyPI: the flashinfer.ai index only redirects to GitHub releases, which time out here.
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=$vllm_base uv pip install .
uv pip install "$tp/openjev" httpx pillow
"$env/bin/python" -c "import torch, vllm; print(torch.__version__, vllm.__version__, torch.cuda.get_device_capability())"
