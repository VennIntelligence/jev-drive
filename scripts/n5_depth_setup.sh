#!/usr/bin/env bash
# Environment + weights for night queue 2, N5 (todos/2026-09-26-night-queue-2.md): camera-conditioned metric depth
# in place of the flat-ground lift, measurement only.
#   envs/depth  torch 2.13 cu130 (same wheels as envs/sam3) + UniDepth (v2) + Depth Anything 3, both from their repos,
#               installed without their pinned extras (xformers is optional in both; app / 3DGS / export deps unused)
# Weights: HF cache ($HF_HOME) via hf-mirror. Idempotent.
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
TP=$DATA_DIR/third_party W=$DATA_DIR/tmp/whl E=$DATA_DIR/envs/depth
MIRROR=https://mirrors.aliyun.com/pypi/simple
(source /etc/network_turbo >/dev/null
 [[ -d $TP/UniDepth/.git ]] || git clone -q --depth 1 https://github.com/lpiccinelli-eth/UniDepth "$TP/UniDepth"
 [[ -d $TP/Depth-Anything-3/.git ]] || git clone -q --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3 "$TP/Depth-Anything-3")
echo "UniDepth $(git -C "$TP/UniDepth" rev-parse HEAD), Depth-Anything-3 $(git -C "$TP/Depth-Anything-3" rev-parse HEAD)"
[[ -x $E/bin/python ]] || uv venv -q --python 3.12 "$E"
if ! "$E/bin/python" -c "import torch" 2>/dev/null; then
  uv pip install -q --python "$E/bin/python" --no-deps "$W"/torch-2.13.0+cu130-*.whl "$W"/torchvision-0.28.0+cu130-*.whl
  deps=$("$E/bin/python" -c "
import importlib.metadata as m
print(' '.join(r.split(';')[0].replace(' ', '') for r in m.requires('torch') if 'extra ==' not in r))")
  uv pip install -q --python "$E/bin/python" $deps pillow --index-url $MIRROR
fi
uv pip install -q --python "$E/bin/python" --index-url $MIRROR einops timm huggingface-hub safetensors omegaconf addict \
  opencv-python-headless imageio scipy pandas pyarrow tqdm tensorboard matplotlib trimesh plyfile evo "numpy<2.3" \
  wandb h5py tables tabulate termcolor protobuf e3nn moviepy==1.0.3 pillow_heif typer requests pycolmap   # import-time deps
uv pip install -q --python "$E/bin/python" --no-deps -e "$TP/UniDepth"
uv pip install -q --python "$E/bin/python" --no-deps -e "$TP/Depth-Anything-3"
# weights into the HF cache (hf-mirror, direct link; the box cannot reach huggingface.co without the proxy)
HF_ENDPOINT=https://hf-mirror.com env -u http_proxy -u https_proxy "$E/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
for r in ("lpiccinelli/unidepth-v2-vitl14", "depth-anything/DA3METRIC-LARGE"):
    print(r, snapshot_download(r))
PY
"$E/bin/python" -c "import unidepth.models, depth_anything_3.api; print('imports ok')"
