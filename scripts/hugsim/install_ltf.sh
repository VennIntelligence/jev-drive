#!/usr/bin/env bash
# Env for HUGSIM's official LTF client (hyzhou404/NAVSIM fork, ltf_e2e.py) on Blackwell. See docs/hugsim.md.
# The fork pins torch 2.0.1+cu118 (no sm_120); we take the package set of our navsim1 env (navsim 1.1 +
# nuplan-devkit 1.2, Python 3.10) and swap torch for 2.8.0+cu128. The LTF code is plain PyTorch, no extensions.
# Usage (on the box): scripts/tmux_run.sh hugsim-ltf-build scripts/hugsim/install_ltf.sh
set -euo pipefail
D=${DATA_DIR:-$HOME/data}
NAV=$D/third_party/hugsim_deps/NAVSIM    # hyzhou404/NAVSIM @ ca0ca7e
ENV=$D/envs/hugsim-ltf
export UV_INDEX_URL=http://mirrors.aliyun.com/pypi/simple UV_INSECURE_HOST=mirrors.aliyun.com
[[ -d $NAV ]] || (source /etc/network_turbo >/dev/null 2>&1; git clone -q https://github.com/hyzhou404/NAVSIM "$NAV")
git -C "$NAV" checkout -q ca0ca7e4368646d8f7b86fb1fdaa1862c946176f
# weights: https://huggingface.co/autonomousvision/navsim_baselines/tree/main/ltf (the path ltf_e2e.py loads)
ck=$NAV/ckpts/ltf_seed_0.ckpt
mkdir -p "$NAV/ckpts"
until [[ $(stat -c %s "$ck" 2>/dev/null || echo 0) == 673235500 ]]; do  # curl resumes; the link drops mid-transfer
  curl -sL --retry 20 -C - -o "$ck" https://hf-mirror.com/autonomousvision/navsim_baselines/resolve/main/ltf/ltf_seed_0.ckpt || sleep 5
done

[[ -x $ENV/bin/python ]] || uv venv "$ENV" --python 3.10
req=$(mktemp)
uv pip freeze --python "$D/envs/navsim1/bin/python" \
  | grep -viE "^(torch|torchvision|triton|nvidia-|navsim|nuplan-devkit)([=@ ]|$)" > "$req"
uv pip install --python "$ENV/bin/python" -r "$req" "torch==2.8.0" "torchvision==0.23.0"
uv pip install --python "$ENV/bin/python" --no-deps "$D/third_party/nuplan-devkit" "$NAV"
rm -f "$req"
"$ENV/bin/python" -c "import torch, navsim, nuplan; a = torch.cuda.get_arch_list(); assert 'sm_120' in a, a; print('ok', torch.__version__)"
