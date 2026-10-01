#!/usr/bin/env bash
# Cosmos pilot, one stage (fc65452:todos/2026-09-28-cosmos-pilot.md): for each variant, Cosmos on the selected pairs (+ the
# x- noise floors; the 35-step seg variant only the other-seed one), then YOLO, pixel metrics, openpilot, and the report.
# Controls must exist (python -m experiments.cosmos.lib.cosmos_pilot controls). GPU 1, scheduler row cosmos-pilot.
# Usage: experiments/cosmos/archive/cosmos_stage.sh <pilot1|all> <variant[,variant...]>     e.g. experiments/cosmos/archive/cosmos_stage.sh pilot1 edgeA,edgeB
# Writes $DATA_DIR/runs/cosmos/stage-<which>.{DONE,ERROR,STATUS}.
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/../../.."
which=$1 variants=$2
E=$DATA_DIR/envs R=$DATA_DIR/runs/cosmos
export CUDA_VISIBLE_DEVICES=${COSMOS_GPU:-1} PYTHONPATH=$PWD
tag=$R/stage-${STAGE_TAG:-$which}
rm -f "$tag.DONE" "$tag.ERROR"
trap 'echo "failed at line $LINENO ($(date +%H:%M))" > "$tag.ERROR"' ERR
st() { echo "$(date '+%m-%d %H:%M') $*" | tee -a "$tag.STATUS"; }
only=$($E/jevdrive/bin/python -c "
import pandas as pd; p = pd.read_csv('experiments/cosmos/results/pairs.csv', dtype={'base_id': str})
from experiments.cosmos.lib.cosmos_pilot import PILOT1
w = '$which'
print(','.join(p[p.base_id == PILOT1].pair if w == 'pilot1' else p.pair if w == 'all' else w.split(',')))")
for v in ${variants//,/ }; do
  # the same-seed re-render is bit-identical (pilot pair, edge/distilled), so the 35-step seg model only gets the
  # other-seed floor
  case $v in seg) model=seg floors=${SEG_FLOORS:-alt} ;; *) model=edge/distilled floors=${FLOORS:-both} ;; esac
  spec=$($E/jevdrive/bin/python -m experiments.cosmos.lib.cosmos_pilot specs --variant "$v" --which "$which" --floors "$floors" | tail -1)
  st "$v: cosmos ($model) on $only"
  $E/cosmos-transfer/bin/python experiments/cosmos/archive/cosmos_infer.py --specs "$spec" --model "$model" --out "$R/out/$v"
  st "$v: detect / pixels / openpilot"
  $E/ultralytics/bin/python -m experiments.cosmos.lib.cosmos_eval detect --variant "$v" --only "$only"
  $E/r3d2/bin/python -m experiments.cosmos.lib.cosmos_eval pixels --variant "$v" --only "$only" --gpu 0
  $E/openpilot/bin/python experiments/cosmos/lib/cosmos_openpilot.py --variant "$v" --pairs "$only"
done
$E/jevdrive/bin/python -m experiments.cosmos.lib.cosmos_eval report --variant "$variants" --only "$only"
st "done"
touch "$tag.DONE"
