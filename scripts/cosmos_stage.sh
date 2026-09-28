#!/usr/bin/env bash
# Cosmos pilot, one stage (todos/2026-09-28-cosmos-pilot.md): for each variant, Cosmos on the selected pairs (+ the
# x- noise floors; the 35-step seg variant only the other-seed one), then YOLO, pixel metrics, openpilot, and the report.
# Controls must exist (python -m jevdrive.cosmos_pilot controls). GPU 1, scheduler row cosmos-pilot.
# Usage: scripts/cosmos_stage.sh <pilot1|all> <variant[,variant...]>     e.g. scripts/cosmos_stage.sh pilot1 edgeA,edgeB
# Writes $DATA_DIR/runs/cosmos/stage-<which>.{DONE,ERROR,STATUS}.
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
which=$1 variants=$2
E=$DATA_DIR/envs R=$DATA_DIR/runs/cosmos
export CUDA_VISIBLE_DEVICES=${COSMOS_GPU:-1} PYTHONPATH=$PWD
tag=$R/stage-$which
rm -f "$tag.DONE" "$tag.ERROR"
trap 'echo "failed at line $LINENO ($(date +%H:%M))" > "$tag.ERROR"' ERR
st() { echo "$(date '+%m-%d %H:%M') $*" | tee -a "$tag.STATUS"; }
only=$($E/jevdrive/bin/python -c "
import pandas as pd; p = pd.read_csv('research/results/cosmos/pairs.csv', dtype={'base_id': str})
from jevdrive.cosmos_pilot import PILOT1
print(','.join(p[p.base_id == PILOT1].pair if '$which' == 'pilot1' else p.pair))")
for v in ${variants//,/ }; do
  # the same-seed re-render is bit-identical (pilot pair, edge/distilled), so the 35-step seg model only gets the
  # other-seed floor
  case $v in seg) model=seg floors=alt ;; *) model=edge/distilled floors=${FLOORS:-both} ;; esac
  spec=$($E/jevdrive/bin/python -m jevdrive.cosmos_pilot specs --variant "$v" --which "$which" --floors "$floors" | tail -1)
  st "$v: cosmos ($model) on $only"
  $E/cosmos-transfer/bin/python scripts/cosmos_infer.py --specs "$spec" --model "$model" --out "$R/out/$v"
  st "$v: detect / pixels / openpilot"
  $E/ultralytics/bin/python -m jevdrive.cosmos_eval detect --variant "$v" --only "$only"
  $E/r3d2/bin/python -m jevdrive.cosmos_eval pixels --variant "$v" --only "$only" --gpu 0
  $E/openpilot/bin/python scripts/cosmos_openpilot.py --variant "$v" --pairs "$only"
done
$E/jevdrive/bin/python -m jevdrive.cosmos_eval report --variant "$variants" --only "$only"
st "done"
touch "$tag.DONE"
