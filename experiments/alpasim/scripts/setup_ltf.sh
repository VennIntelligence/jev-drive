#!/usr/bin/env bash
# Env and checkpoint for the shipped SimScale Latent TransFuser sample driver, run natively instead of in its
# Docker image. Deviation from its requirements.txt: torch 2.8.0 / torchvision 0.23.0 instead of 2.6.0 / 0.21.0
# (the RTX 6000D is sm_120, which the cu124 build of torch 2.6 does not target).
# Usage (box): scripts/tmux_run.sh alpasim-ltf experiments/alpasim/scripts/setup_ltf.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
SRC=${ALPASIM_SRC:-$DATA_DIR/third_party/alpasim}
S=$SRC/e2e_challenge/sample_submission_simscale_navsim_transfuser
ENV=$DATA_DIR/envs/alpasim-ltf; CK=$DATA_DIR/models/alpasim_ltf
OUT=$DATA_DIR/runs/alpasim/setup_ltf; mkdir -p "$OUT" "$CK"; rm -f "$OUT/DONE" "$OUT/ERROR"
exec > >(tee -a "$OUT/log.txt") 2>&1
trap 'echo "failed at line $LINENO" > "$OUT/ERROR"' ERR
PIP="uv pip install --python $ENV/bin/python --default-index https://mirrors.aliyun.com/pypi/simple"

[[ -x $ENV/bin/python ]] || uv venv --python 3.12 "$ENV"
sed 's/^torchvision==.*/torchvision==0.23.0/' "$S/requirements.txt" > "$OUT/requirements.txt"
$PIP torch==2.8.0 -r "$OUT/requirements.txt"
$PIP --no-deps "$SRC/src/grpc"
if [[ ! -e $CK/ltf_sim_navtest.ckpt ]]; then
  curl -L --retry 5 -o "$CK/dl.ckpt" \
    https://hf-mirror.com/datasets/OpenDriveLab/SimScale/resolve/main/SimScale_ckpts/LTF/ltf_sim_navtest.ckpt
  LTF_ASSET_DIR=$CK bash "$S/scripts/prepare_assets.sh" "$CK/dl.ckpt" && rm -f "$CK/dl.ckpt"   # checks size + sha256
fi
"$ENV/bin/python" -c "import torch, torchvision, alpasim_grpc, timm, numpy; print('torch', torch.__version__, 'tv', torchvision.__version__, 'timm', timm.__version__, 'numpy', numpy.__version__, 'grpc api', alpasim_grpc.__version__, 'archs', torch.cuda.get_arch_list())" | tee "$OUT/versions.txt"
du -sh "$ENV" "$CK" | tee -a "$OUT/versions.txt"
date > "$OUT/DONE"
