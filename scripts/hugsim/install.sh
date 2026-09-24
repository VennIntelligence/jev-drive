#!/usr/bin/env bash
# Build the HUGSIM simulation env on a Blackwell (sm_120) box. See docs/hugsim.md.
# Stack: Python 3.12, torch 2.8.0 + cu128 (PyPI build), nvcc 12.8, TORCH_CUDA_ARCH_LIST=12.0.
# Only the simulation/eval path is built; training-only deps (pytorch3d, unidepth, apex, ...) are skipped.
# Usage (on the box, CPU only): scripts/tmux_run.sh hugsim-build scripts/hugsim/install.sh
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
D=${DATA_DIR:-$HOME/data}
HUG=$D/third_party/HUGSIM          # hyzhou404/HUGSIM @ 62c690d
DEPS=$D/third_party/hugsim_deps    # HUGSIM_splat, tiny-cuda-nn, trajdata, nuscenes-devkit
ENV=$D/envs/hugsim
export UV_INDEX_URL=http://mirrors.aliyun.com/pypi/simple UV_INSECURE_HOST=mirrors.aliyun.com
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export TORCH_CUDA_ARCH_LIST=12.0 TCNN_CUDA_ARCHITECTURES=120 MAX_JOBS=${MAX_JOBS:-8}
export VIRTUAL_ENV=$ENV
pip() { uv pip install --python "$ENV/bin/python" "$@"; }

clone() {  # clone <github repo> <dir> <commit>
  [[ -d $2 ]] || (source /etc/network_turbo >/dev/null 2>&1; git clone -q --recursive "https://github.com/$1" "$2")
  git -C "$2" checkout -q "$3" && git -C "$2" submodule update -q --init --recursive
}
clone hyzhou404/HUGSIM "$HUG" 62c690d39fd90020e68a196bd8bcc1c4d4191f2e
clone hyzhou404/HUGSIM_splat "$DEPS/HUGSIM_splat" 88f2a40c4e2f6bafde2beeaba6c43bbb0ccb1f5f
clone NVlabs/tiny-cuda-nn "$DEPS/tiny-cuda-nn" 0109538c37ac0bf613f2bac8de6cda48352feca7
clone hyzhou404/trajdata "$DEPS/trajdata" dea018df54dd4917165429932ec4f8b645e04f07
clone hyzhou404/nuscenes-devkit "$DEPS/nuscenes-devkit" adf6972147a474185915fd68cae4ccb2f0355957

# our patches to HUGSIM (idempotent)
for p in "$repo"/patches/hugsim/*.patch; do
  git -C "$HUG" apply --reverse --check "$p" 2>/dev/null || git -C "$HUG" apply "$p"
done

[[ $("$ENV/bin/python" -V 2>/dev/null) == "Python 3.12"* ]] || uv venv --clear "$ENV" --python 3.12
echo "== torch + pure-python deps"
pip "torch==2.8.0" "torchvision==0.23.0" "numpy>=1.26,<2" "setuptools<80" wheel ninja \
  scipy opencv-python matplotlib tqdm "roma>=1.5" "open3d==0.19.0" "gymnasium==1.1.1" "omegaconf>=2.3" \
  "shapely>=2.1" plyfile pyyaml "imageio>=2.37" "moviepy>=2.2.1,<3" "jaxtyping>=0.3.1" rich einops \
  pyquaternion tabulate "pillow>=11.2" "protobuf==3.20.2" timm trimesh h5py \
  pandas pyarrow zarr kornia seaborn bokeh geopandas dill descartes cachetools fire
"$ENV/bin/python" -c "import torch; a = torch.cuda.get_arch_list(); assert 'sm_120' in a, a"

echo "== from source (no build isolation, against the torch above)"
pip --no-build-isolation "$HUG/submodules/simple-knn"
pip --no-build-isolation "$DEPS/HUGSIM_splat"
pip --no-build-isolation "$DEPS/tiny-cuda-nn/bindings/torch"
pip --no-deps "$DEPS/trajdata" "$DEPS/nuscenes-devkit"
pip -e "$HUG/sim"

echo "== import check"
cd "$HUG" && "$ENV/bin/python" - <<'EOF'
import sys, torch; sys.path.insert(0, '.')
import gsplat, tinycudann, simple_knn._C, trajdata, hugsim_env, open3d
from gaussian_renderer import render
from sim.utils.score_calculator import hugsim_evaluate
print('ok', torch.__version__, torch.version.cuda, torch.cuda.get_arch_list(), 'gsplat', gsplat.__version__)
EOF
