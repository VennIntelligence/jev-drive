#!/usr/bin/env bash
# ControlNet pair pilot (research/world-model/index.html), one stage.
# Usage: experiments/controlnet_pair/archive/cn_pair.sh <tag> <scenes: comma list or all> <arms, comma list, in order; EG / EI / EIG after E, BVI after BV>
# SCENES_<ARM>=<list> overrides the scene list for one arm (e.g. SCENES_BV=p3_000).
# GPU from CN_GPU (default 2). Writes $DATA_DIR/runs/cn_pair/<tag>.{DONE,ERROR,STATUS}.
set -Eeuo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/../../.."
tag=$1 scenes=$2 arms=$3
E=$DATA_DIR/envs R=$DATA_DIR/runs/cn_pair
mkdir -p "$R"
export CUDA_VISIBLE_DEVICES=${CN_GPU:-2} PYTHONPATH=$PWD
T=$R/$tag
rm -f "$T.DONE" "$T.ERROR"
trap 'echo "failed at line $LINENO ($(date +%H:%M))" > "$T.ERROR"' ERR
st() { echo "$(date '+%m-%d %H:%M') $*" | tee -a "$T.STATUS"; }
cn() { "$E/$1/bin/python" -m experiments.controlnet_pair.archive.cn_pair "${@:2}"; }
free_gb() { echo $(( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES") / 1024 )); }
infer() {  # infer <spec> <model> <arm>: one Cosmos process at a time (lock), and only when the card has room (75 GB cap)
  local need=$([[ $2 == edge/distilled ]] && echo 45 || echo 58)
  (
    flock 9
    until (( $(free_gb) >= need + 9 )); do sleep 60; done
    "$E/cosmos-transfer/bin/python" experiments/controlnet_pair/archive/cn_pair_infer.py --specs "$1" --model "$2" --out "$R/gen/$3"
  ) 9> "$R/gpu.lock"
}
st "prep $scenes"
cn ultralytics prep --scenes "$scenes"
for arm in ${arms//,/ }; do
  ov=SCENES_$arm; scenes=${!ov:-$2}
  case $arm in
    E|Ed) st "$arm"; infer "$(cn jevdrive specs --arm "$arm" --scenes "$scenes" --tag "_$tag" | tail -1)" edge/distilled "$arm" ;;
    EG)   st "EG guided x+ on E's x-"; cn jevdrive anchor --arm E --scenes "$scenes"
          infer "$(cn jevdrive specs --arm EG --scenes "$scenes" --tag "_$tag" | tail -1)" edge/distilled EG
          cn jevdrive blend --arm EG --base E --scenes "$scenes"; arm=EGb ;;
    B|BV) st "$arm (base, 35 steps)"; infer "$(cn jevdrive specs --arm "$arm" --scenes "$scenes" --tag "_$tag" | tail -1)" edge "$arm" ;;
    EI|BVI) st "$arm insertion: depth, walker, x+"; cn depth depth --scenes "$scenes"; cn jevdrive insert --scenes "$scenes"
          infer "$(cn jevdrive specs --arm "$arm" --scenes "$scenes" --tag "_$tag" | tail -1)" \
                "$([[ $arm == EI ]] && echo edge/distilled || echo edge)" "$arm"
          cn jevdrive link --arm "$arm" --scenes "$scenes" ;;
    EIG)  st "EIG guided insertion on E's x-"; cn jevdrive anchor --arm E --scenes "$scenes"
          infer "$(cn jevdrive specs --arm EIG --scenes "$scenes" --tag "_$tag" | tail -1)" edge/distilled EIG
          cn jevdrive blend --arm EIG --base E --scenes "$scenes"; arm=EIGb ;;
  esac
  st "webp $arm"; cn jevdrive webp --arm "$arm" --scenes "$scenes"
done
st done
touch "$T.DONE"
