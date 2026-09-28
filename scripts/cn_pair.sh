#!/usr/bin/env bash
# ControlNet pair pilot (research/controlnet-pair-pilot.md), one stage.
# Usage: scripts/cn_pair.sh <tag> <scenes: comma list or all> <arms, comma list, in order; EG after E>
# GPU from CN_GPU (default 2). Writes $DATA_DIR/runs/cn_pair/<tag>.{DONE,ERROR,STATUS}.
set -Eeuo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
tag=$1 scenes=$2 arms=$3
E=$DATA_DIR/envs R=$DATA_DIR/runs/cn_pair
mkdir -p "$R"
export CUDA_VISIBLE_DEVICES=${CN_GPU:-2} PYTHONPATH=$PWD
T=$R/$tag
rm -f "$T.DONE" "$T.ERROR"
trap 'echo "failed at line $LINENO ($(date +%H:%M))" > "$T.ERROR"' ERR
st() { echo "$(date '+%m-%d %H:%M') $*" | tee -a "$T.STATUS"; }
cn() { "$E/$1/bin/python" -m jevdrive.cn_pair "${@:2}"; }
free_gb() { echo $(( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES") / 1024 )); }
infer() {  # infer <spec> <model> <arm>: waits until the card has room under the 75 GB cap
  local need=$([[ $2 == edge/distilled ]] && echo 45 || echo 58)
  until (( $(free_gb) >= need + 9 )); do sleep 60; done
  "$E/cosmos-transfer/bin/python" scripts/cn_pair_infer.py --specs "$1" --model "$2" --out "$R/gen/$3"
}
st "prep $scenes"
cn ultralytics prep --scenes "$scenes"
for arm in ${arms//,/ }; do
  case $arm in
    E|Ed) st "$arm"; infer "$(cn jevdrive specs --arm "$arm" --scenes "$scenes" --tag "_$tag" | tail -1)" edge/distilled "$arm" ;;
    EG)   st "EG guided x+ on E's x-"; cn jevdrive anchor --arm E --scenes "$scenes"
          infer "$(cn jevdrive specs --arm EG --scenes "$scenes" --tag "_$tag" | tail -1)" edge/distilled EG
          cn jevdrive blend --arm EG --base E --scenes "$scenes"; arm=EGb ;;
    B|BV) st "$arm (base, 35 steps)"; infer "$(cn jevdrive specs --arm "$arm" --scenes "$scenes" --tag "_$tag" | tail -1)" edge "$arm" ;;
  esac
  st "webp $arm"; cn jevdrive webp --arm "$arm" --scenes "$scenes"
done
st done
touch "$T.DONE"
