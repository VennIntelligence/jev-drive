#!/usr/bin/env bash
# Build the drivestudio (OmniRe) envs on a Blackwell (sm_120) box for nq4 P3 (todos/2026-09-26-night-queue-4.md, P section).
# Upstream pins torch 2.0.0+cu117 (no sm_120 kernels); we follow the HUGSIM recipe (docs/hugsim.md):
#   envs/drivestudio : Python 3.10, torch 2.8.0+cu128 (PyPI), nvcc 12.8, gsplat v1.3.0 + pytorch3d built for sm_120.
#   envs/p3-wodprep  : Python 3.10, tensorflow 2.12 + waymo-open-dataset-tf-2-12-0, only for drivestudio's Waymo preprocess.
# CPU only. Usage (on the box): taskset -c <8 cores> scripts/p3/install_drivestudio.sh
set -euo pipefail
D=${DATA_DIR:-$HOME/data}
DS=$D/third_party/drivestudio            # ziyc/drivestudio @ e59bda4
ENV=$D/envs/drivestudio
PREP=$D/envs/p3-wodprep
DEPS=$D/third_party/p3_deps
export UV_INDEX_URL=http://mirrors.aliyun.com/pypi/simple UV_INSECURE_HOST=mirrors.aliyun.com
export UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=${MAX_JOBS:-8} FORCE_CUDA=1
pip() { local py=$1; shift; uv pip install --python "$py/bin/python" "$@"; }

mkdir -p "$DEPS"
clone() {  # clone <github repo> <dir> <tag or commit>; shallow for tags, turbo first, then Clash
  if [[ ! -d $2 ]]; then
    local c=(git clone -q --recursive --shallow-submodules --depth 1 --branch "$3" "https://github.com/$1" "$2")
    (source /etc/network_turbo >/dev/null 2>&1; "${c[@]}") || { rm -rf "$2"; bash -lc "proxy_on >/dev/null 2>&1; $(printf '%q ' "${c[@]}")"; }
  fi
  git -C "$2" checkout -q "$3" 2>/dev/null || true
}
[[ -d $DS ]] || { git clone -q --recursive https://github.com/ziyc/drivestudio "$DS"; git -C "$DS" checkout -q e59bda4; }
clone nerfstudio-project/gsplat "$DEPS/gsplat" v1.3.0
clone facebookresearch/pytorch3d "$DEPS/pytorch3d" V0.7.8
clone NVlabs/nvdiffrast "$DEPS/nvdiffrast" v0.3.3

[[ $("$ENV/bin/python" -V 2>/dev/null) == "Python 3.10"* ]] || uv venv --clear "$ENV" --python 3.10
echo "== drivestudio: torch + pure-python deps"
pip "$ENV" "torch==2.8.0" "torchvision==0.23.0" "numpy>=1.26,<2" "setuptools<80" wheel ninja \
  "timm>=0.9.5" "pytorch_msssim==1.0.0" "omegaconf==2.3.0" "torchmetrics>=1.0,<1.5" tensorboard wandb \
  matplotlib plotly "viser==0.2.1" "nerfview==0.0.3" imageio imageio-ffmpeg scikit-image opencv-python \
  "open3d==0.19.0" "pyquaternion==0.9.9" "kornia==0.7.2" tqdm gdown "lpips==0.1.4" trimesh \
  nuscenes-devkit "transformers>=4.44,<5" pandas pyarrow jaxtyping rich
"$ENV/bin/python" -c "import torch; a = torch.cuda.get_arch_list(); assert 'sm_120' in a, a"

echo "== drivestudio: CUDA extensions from source (no build isolation, against the torch above)"
"$ENV/bin/python" -c "import gsplat" 2>/dev/null || pip "$ENV" --no-build-isolation "$DEPS/gsplat"
"$ENV/bin/python" -c "import pytorch3d._C" 2>/dev/null || pip "$ENV" --no-build-isolation "$DEPS/pytorch3d"
"$ENV/bin/python" -c "import nvdiffrast.torch" 2>/dev/null || pip "$ENV" --no-build-isolation "$DEPS/nvdiffrast"
pip "$ENV" -e "$DS/third_party/smplx"

echo "== drivestudio: import check"
cd "$DS" && PYTHONPATH=$DS "$ENV/bin/python" - <<'PY'
import torch, gsplat, pytorch3d.ops, open3d, nuscenes, transformers
from models.trainers import MultiTrainer
x = torch.randn(1000, 3, device='cuda'); pytorch3d.ops.knn_points(x[None], x[None], K=3)
print('ok', torch.__version__, torch.version.cuda, 'gsplat', gsplat.__version__)
PY

echo "== p3-wodprep: tensorflow + waymo-open-dataset (Waymo preprocess only)"
[[ $("$PREP/bin/python" -V 2>/dev/null) == "Python 3.10"* ]] || uv venv --clear "$PREP" --python 3.10
pip "$PREP" "waymo-open-dataset-tf-2-12-0==1.6.4" "protobuf==3.20.3" "numpy<2" pillow tqdm opencv-python-headless
CUDA_VISIBLE_DEVICES= "$PREP/bin/python" -c "import tensorflow as tf; from waymo_open_dataset.utils import frame_utils; print('ok tf', tf.__version__)"
echo "== install done"
