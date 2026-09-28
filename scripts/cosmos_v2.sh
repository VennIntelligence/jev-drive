#!/usr/bin/env bash
# Cosmos pilot v2 (todos/2026-09-28-cosmos-pilot.md, "v2"), one stage. GPU 1, scheduler row cosmos-pilot, cores 24-47.
# Usage: scripts/cosmos_v2.sh <tag> <pairs (comma list or all)> <arms: E2,G2,P2,M2 subset> [M2 pairs]
# Writes $DATA_DIR/runs/cosmos/v2-<tag>.{DONE,ERROR,STATUS}.
set -Eeuo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
tag=$1 which=$2 arms=$3 mpairs=${4:-$2}
E=$DATA_DIR/envs R=$DATA_DIR/runs/cosmos
export CUDA_VISIBLE_DEVICES=${COSMOS_GPU:-1} PYTHONPATH=$PWD
T=$R/v2-$tag
rm -f "$T.DONE" "$T.ERROR"
trap 'echo "failed at line $LINENO ($(date +%H:%M))" > "$T.ERROR"' ERR
st() { echo "$(date '+%m-%d %H:%M') $*" | tee -a "$T.STATUS"; }
py() { $E/jevdrive/bin/python -m jevdrive.cosmos_v2 "$@"; }
free_gb() { echo $(( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES") / 1024 )); }
infer() {  # infer <spec> <model> <arm>: waits until the card has room
  local need=$([[ $2 == edge/distilled ]] && echo "${NEED_DISTILLED:-36}" || echo "${NEED_BASE:-52}")   # text encoder on CPU
  until (( $(free_gb) >= need )); do sleep 60; done
  $E/cosmos-transfer/bin/python scripts/cosmos_infer.py --specs "$1" --model "$2" --out "$R/out/$3"
}
has() { [[ ",$arms," == *",$1,"* ]]; }
st "controls2 on $which"
[[ ${SKIP_CONTROLS:-0} == 1 ]] || py controls2 --which "$which"
eval_arms=()
if [[ ${EVAL_ONLY:-} ]]; then read -ra eval_arms <<< "${EVAL_ONLY//,/ }"; arms=""; fi
if has E2; then st "E2"; infer "$(py specs2 --arm E2 --which "$which" | tail -1)" edge/distilled E2; eval_arms+=(E2); fi
if has P2; then st "P2 (x- only, prompt ablation)"; infer "$(py specs2 --arm P2 --which "$which" --members minus --no-floors | tail -1)" edge/distilled P2; fi
if has G2; then
  st "G2 guided x+"; py anchor --arm E2 --which "$which"
  infer "$(py specs2 --arm G2 --which "$which" | tail -1)" edge/distilled G2
  py blend --which "$which" --arm G2; eval_arms+=(G2b)
fi
if has E3; then st "E3"; infer "$(py specs2 --arm E3 --which "$which" | tail -1)" edge/distilled E3; eval_arms+=(E3); fi
if has G3; then
  st "G3 guided x+"; py anchor --arm E3 --which "$which"
  infer "$(py specs2 --arm G3 --which "$which" | tail -1)" edge/distilled G3
  py blend --which "$which" --arm G3; eval_arms+=(G3b)
fi
if has E4; then st "E4"; infer "$(py specs2 --arm E4 --which "$which" | tail -1)" edge/distilled E4; eval_arms+=(E4); fi
if has G4; then
  st "G4 guided x+"; py anchor --arm E4 --which "$which"
  infer "$(py specs2 --arm G4 --which "$which" | tail -1)" edge/distilled G4
  py blend --which "$which" --arm G4; eval_arms+=(G4b)
fi
if has M2; then st "M2 multicontrol base on $mpairs"; infer "$(py specs2 --arm M2 --which "$mpairs" --no-floors | tail -1)" seg M2; eval_arms+=(M2); fi
for v in "${eval_arms[@]}"; do
  only=$([[ $v == M2 ]] && echo "$mpairs" || echo "$which"); [[ $only == all ]] && only=""
  st "eval $v"
  $E/ultralytics/bin/python -m jevdrive.cosmos_eval detect --variant "$v" --only "$only"
  $E/r3d2/bin/python -m jevdrive.cosmos_eval pixels --variant "$v" --only "$only" --gpu 0
  $E/openpilot/bin/python scripts/cosmos_openpilot.py --variant "$v" ${only:+--pairs "$only"}
done
IFS=, ; va="${eval_arms[*]}"; unset IFS
$E/jevdrive/bin/python -m jevdrive.cosmos_eval report --variant "$va" --only "$([[ $which == all ]] || echo "$which")"
$E/jevdrive/bin/python -m jevdrive.cosmos_white --variants "$va" --tag "_v2_$tag"
st done
touch "$T.DONE"
