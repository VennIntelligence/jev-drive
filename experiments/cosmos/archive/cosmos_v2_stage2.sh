#!/usr/bin/env bash
# Cosmos v2 stage 2 (frozen arm G4): E4 x- + seed floor, G4 guided x+ on the 6 pairs not in stage 1, then G4b on all 10.
set -Eeuo pipefail
export DATA_DIR=/root/autodl-tmp/ujs; cd $DATA_DIR/jev-drive
E=$DATA_DIR/envs R=$DATA_DIR/runs/cosmos T=$R/v2-s2
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD
rm -f $T.DONE $T.ERROR; trap 'echo "failed at line $LINENO ($(date +%H:%M))" > $T.ERROR' ERR
st() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $T.STATUS; }
py() { $E/jevdrive/bin/python -m experiments.cosmos.archive.cosmos_v2 "$@"; }
infer() { until (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1) / 1024 >= 45 )); do sleep 60; done
          $E/cosmos-transfer/bin/python experiments/cosmos/archive/cosmos_infer.py --specs "$1" --model edge/distilled --out "$R/out/$2"; }
NEW=24206-s0,24224-s0,24294-s0,24519-s0,27297-s0,27515-s0
st "E4 x- and seed floor on $NEW"; infer "$(py specs2 --arm E4 --which $NEW --members minus | tail -1)" E4
st "G4 guided x+"; py anchor --arm E4 --which $NEW; infer "$(py specs2 --arm G4 --which $NEW | tail -1)" G4
py blend --which $NEW --arm G4
st "eval G4b (10 pairs)"
$E/ultralytics/bin/python -m experiments.cosmos.lib.cosmos_eval detect --variant G4b --only $NEW
$E/r3d2/bin/python -m experiments.cosmos.lib.cosmos_eval pixels --variant G4b --only $NEW --gpu 0
$E/openpilot/bin/python experiments/cosmos/lib/cosmos_openpilot.py --variant G4b --pairs $NEW
$E/jevdrive/bin/python -m experiments.cosmos.lib.cosmos_eval report --variant G4b,edgeB --only ""
$E/jevdrive/bin/python -m experiments.cosmos.archive.cosmos_white --variants G4b,edgeB --tag _v2_final
st done; touch $T.DONE
