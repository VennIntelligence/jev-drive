#!/usr/bin/env bash
# Environments + weights for the fast-perception comparison (todos/2026-09-26-fast-perception.md).
#   envs/efficientsam3  EfficientSAM3 (its own fork of the sam3 package; must not go into envs/sam3)
#   envs/ultralytics    YOLOE-26 / YOLO26 (-seg, -depth)
#   envs/gdino          Grounding DINO tiny via HF transformers
# Weights land in $DATA_DIR/models/{efficientsam3,ultralytics,grounding-dino-tiny}. Idempotent.
# Network: HF files through hf-mirror (jevdrive.hfdl, direct), GitHub through network_turbo, PyPI through the Aliyun mirror.
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
REPO=$PWD M=$DATA_DIR/models TP=$DATA_DIR/third_party W=$DATA_DIR/tmp/whl
MIRROR=https://mirrors.aliyun.com/pypi/simple
JPY=$DATA_DIR/envs/jevdrive/bin/python

hf() {  # repo path dst
  [[ -s $3 ]] || env -u http_proxy -u https_proxy PYTHONPATH=$REPO "$JPY" -c "
import sys; from pathlib import Path; from jevdrive.hfdl import download, hf_url
download(hf_url(sys.argv[1], sys.argv[2]), Path(sys.argv[3]), streams=16)" "$1" "$2" "$3"
}
gh_asset() {  # name dst_dir; GitHub release assets through turbo, resumed across dropped connections
  [[ -s $2/$1 ]] && return
  (source /etc/network_turbo >/dev/null
   for i in 1 2 3 4 5 6 7 8; do
     curl -fsSL -C - -o "$2/$1.tmp" "https://github.com/ultralytics/assets/releases/download/v8.4.0/$1" && break
   done) && mv "$2/$1.tmp" "$2/$1"
}
torch_env() {  # env dir: python 3.12 + torch 2.13 cu130 from the local wheel cache (same as envs/sam3)
  [[ -x $1/bin/python ]] || uv venv -q --python 3.12 "$1"
  "$1/bin/python" -c "import torch" 2>/dev/null && return
  uv pip install -q --python "$1/bin/python" --no-deps "$W"/torch-2.13.0+cu130-*.whl "$W"/torchvision-0.28.0+cu130-*.whl
  local deps
  deps=$("$1/bin/python" -c "
import importlib.metadata as m
print(' '.join(r.split(';')[0].replace(' ', '') for r in m.requires('torch') if 'extra ==' not in r))")
  uv pip install -q --python "$1/bin/python" $deps pillow --index-url $MIRROR
}
common="pandas pyarrow pycocotools scipy tqdm tensorboard psutil opencv-python-headless"

# ---- EfficientSAM3
E=$DATA_DIR/envs/efficientsam3
[[ -d $TP/efficientsam3/.git ]] || (source /etc/network_turbo >/dev/null; git clone -q --depth 1 https://github.com/SimonZeng7108/efficientsam3 "$TP/efficientsam3")
echo "efficientsam3 code at $(git -C "$TP/efficientsam3" rev-parse HEAD)"
torch_env "$E"
# no [stage1] extra: it pulls mmcv (a source build, training only); inference needs the core deps below
"$E/bin/python" -c "import sam3.model_builder" 2>/dev/null || uv pip install -q --python "$E/bin/python" --index-url $MIRROR \
  -e "$TP/efficientsam3" $common einops timm omegaconf "setuptools<81"
"$E/bin/python" -c "import sam3.model_builder" 2>/dev/null || uv pip install -q --python "$E/bin/python" --index-url $MIRROR \
  -e "$TP/efficientsam3/sam3"
mkdir -p "$M/efficientsam3"
for f in efficientvit repvit tinyvit; do hf Simon7108528/EfficientSAM3 "efficientsam3_ft/efficientsam3_$f.pt" "$M/efficientsam3/efficientsam3_$f.pt"; done

# ---- Ultralytics
U=$DATA_DIR/envs/ultralytics
torch_env "$U"
"$U/bin/python" -c "import ultralytics" 2>/dev/null || uv pip install -q --python "$U/bin/python" --index-url $MIRROR ultralytics $common
mkdir -p "$M/ultralytics"
for f in yolo26x-seg.pt yolo26l-seg.pt; do hf Ultralytics/YOLO26 "$f" "$M/ultralytics/$f"; done   # HF copy, via hf-mirror
for f in yoloe-26x-seg.pt yoloe-26l-seg.pt mobileclip2_b.ts yolo26x-depth.pt; do gh_asset "$f" "$M/ultralytics"; done
# YOLOE's first set_classes() pip-installs ultralytics/CLIP from GitHub; do it here, through turbo, once.
"$U/bin/python" -c "import clip" 2>/dev/null || (source /etc/network_turbo >/dev/null
  uv pip install -q --python "$U/bin/python" "git+https://github.com/ultralytics/CLIP.git")

# ---- Grounding DINO tiny
G=$DATA_DIR/envs/gdino
torch_env "$G"
"$G/bin/python" -c "import transformers" 2>/dev/null || uv pip install -q --python "$G/bin/python" --index-url $MIRROR transformers $common
mkdir -p "$M/grounding-dino-tiny"
for f in config.json model.safetensors preprocessor_config.json special_tokens_map.json tokenizer.json tokenizer_config.json vocab.txt; do
  hf IDEA-Research/grounding-dino-tiny "$f" "$M/grounding-dino-tiny/$f"
done

for e in "$E" "$U" "$G"; do "$e/bin/python" -c "import torch; print('$(basename "$e")', torch.__version__, torch.cuda.is_available())"; done
du -sh "$M/efficientsam3" "$M/ultralytics" "$M/grounding-dino-tiny"
