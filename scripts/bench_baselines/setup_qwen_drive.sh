#!/usr/bin/env bash
# Build the Qwen-Drive-1.0 venv on the box: code in $DATA_DIR/third_party/qwen-drive, venv in $DATA_DIR/envs/qwen-drive.
# Follows the repo's requirements.txt (torch 2.8.0 cu128 already has sm_120 kernels, so no version change was needed).
set -euo pipefail
: "${DATA_DIR:?}"
src=$DATA_DIR/third_party/qwen-drive env=$DATA_DIR/envs/qwen-drive
commit=28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0
[[ -d $src ]] || git clone https://github.com/QwenLM/Qwen-Drive-1.0 "$src"
git -C "$src" checkout -q $commit
[[ -x $env/bin/python ]] || uv venv -p 3.12 "$env"
# Tsinghua mirror: ~2 MB/s per connection here vs ~1 for Aliyun; few connections so dataset downloads are not starved.
export VIRTUAL_ENV=$env UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple UV_CONCURRENT_DOWNLOADS=3 MAX_JOBS=8
uv pip install torch==2.8.0 torchvision==0.23.0 setuptools wheel ninja packaging psutil
uv pip install -r <(grep -v -E '^(flash-attn|causal-conv1d)' "$src/requirements.txt") pyarrow
# flash-attn and causal-conv1d fetch prebuilt wheels from GitHub releases in setup.py; turbo speeds that up.
set +u; source /etc/network_turbo >/dev/null; set -u
uv pip install --no-build-isolation flash-attn==2.8.3 causal-conv1d==1.6.2.post1
unset http_proxy https_proxy
uv pip install --no-build-isolation --no-deps -e "$src"
"$env/bin/python" -c "import torch, flash_attn, causal_conv1d, fla, qwen_drive; print(torch.__version__, torch.cuda.get_device_capability(), flash_attn.__version__)"
