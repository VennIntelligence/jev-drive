#!/usr/bin/env bash
# Insertion-tool comparison (research/insertion-options.md): envs and weights for the two open tools we try.
#   R3D2  (zenseact/R3D2 @ bdd2b4b, code Apache-2.0, weights bertaveira/R3D2{,-big} non-commercial / Waymo licence):
#         one-step SD-Turbo harmonizer trained on Waymo FRONT 3DGS renders with inserted assets (adds shadow + relight).
#   VACE  (ali-vilab/VACE @ 48eb44f + Wan2.1, Apache-2.0; weights Wan-AI/Wan2.1-VACE-14B from ModelScope):
#         masked video-to-video editing, used as a temporally consistent local harmonizer.
# CPU / network only. Envs under $DATA_DIR/envs/{r3d2,vace}; weights under $DATA_DIR/models/{r3d2,vace}.
#   scripts/tmux_run.sh ins-install taskset -c 172-177 scripts/p3/install_insert_tools.sh
set -euo pipefail
D=${DATA_DIR:-$HOME/data}
T=$D/third_party/ins
TORCH="torch==2.8.0 torchvision==0.23.0"                 # PyPI torch 2.8.0 is the cu128 build (sm_120 kernels)
export UV_INDEX_URL=http://mirrors.aliyun.com/pypi/simple UV_INSECURE_HOST=mirrors.aliyun.com
export UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone
mkdir -p "$T" "$D/models/r3d2" "$D/models/vace"
turbo() { (source /etc/network_turbo >/dev/null 2>&1; "$@"); }   # GitHub / HF only; PyPI goes to the mirror

clone() {  # repo dir commit
  [[ -d $T/$2 ]] || turbo git clone -q "$1" "$T/$2"
  git -C "$T/$2" checkout -q "$3"
}
clone https://github.com/zenseact/R3D2 R3D2 bdd2b4b
clone https://github.com/ali-vilab/VACE VACE 48eb44f
clone https://github.com/Wan-Video/Wan2.1 Wan2.1 main

# --- R3D2 env -------------------------------------------------------------------------------------------------------
if [[ ! -f $D/envs/r3d2/DONE ]]; then
  uv venv -q --clear --python 3.11 "$D/envs/r3d2"
  VIRTUAL_ENV=$D/envs/r3d2 uv pip install -q $TORCH
  VIRTUAL_ENV=$D/envs/r3d2 uv pip install -q "diffusers==0.35.1" "transformers>=4.49,<5" "accelerate>=1.4" tyro peft \
    lpips torchmetrics scipy pillow imageio imageio-ffmpeg huggingface_hub
  VIRTUAL_ENV=$D/envs/r3d2 uv pip install -q --no-deps -e "$T/R3D2"
  touch "$D/envs/r3d2/DONE"
fi

# --- VACE env (Wan2.1 backend; SDPA attention, no flash-attn build needed) ---------------------------------------------
if [[ ! -f $D/envs/vace/DONE ]]; then
  uv venv -q --clear --python 3.11 "$D/envs/vace"
  VIRTUAL_ENV=$D/envs/vace uv pip install -q $TORCH
  VIRTUAL_ENV=$D/envs/vace uv pip install -q "diffusers>=0.31" "transformers>=4.49,<5" "tokenizers>=0.20.3" "accelerate>=1.1.1" \
    "opencv-python-headless>=4.9" "numpy>=1.23.5,<2" tqdm imageio imageio-ffmpeg easydict ftfy decord einops scikit-image \
    scipy pillow dashscope modelscope
  VIRTUAL_ENV=$D/envs/vace uv pip install -q --no-deps -e "$T/Wan2.1"
  touch "$D/envs/vace/DONE"
fi

# --- weights ---------------------------------------------------------------------------------------------------------
export HF_TOKEN=${HF_TOKEN:-$(cat "$D/cache/huggingface/token" 2>/dev/null || true)}
for m in R3D2 R3D2-big; do
  [[ -f $D/models/r3d2/$m/DONE ]] || { turbo "$D/envs/r3d2/bin/hf" download "bertaveira/$m" --local-dir "$D/models/r3d2/$m" && touch "$D/models/r3d2/$m/DONE"; }
done
for m in stabilityai/sd-turbo madebyollin/taesd; do turbo "$D/envs/r3d2/bin/hf" download "$m" >/dev/null; done
if [[ ! -f $D/models/vace/Wan2.1-VACE-14B/DONE ]]; then
  "$D/envs/vace/bin/modelscope" download --model Wan-AI/Wan2.1-VACE-14B --local_dir "$D/models/vace/Wan2.1-VACE-14B"
  touch "$D/models/vace/Wan2.1-VACE-14B/DONE"
fi
"$D/envs/r3d2/bin/python" -c "import torch, diffusers, r3d2; print('r3d2 env ok', torch.__version__, diffusers.__version__)"
"$D/envs/vace/bin/python" -c "import sys; sys.path[:0] = ['$T/VACE', '$T/VACE/vace']; import torch, wan; from models.wan import WanVace; print('vace env ok', torch.__version__)"
echo INSTALL_DONE
