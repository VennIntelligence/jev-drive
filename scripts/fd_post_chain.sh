#!/usr/bin/env bash
# fusion post-batch chain: waits for the SAM batch, then Q4d latency (GPU 0, alone), Q8, nuScenes report, Q2b
set -euo pipefail
S=$DATA_DIR/processed/fusion_diag/sam L=$DATA_DIR/processed/fusion_diag/lists
until [ "$(ls $S/nusc/part-*.parquet 2>/dev/null | wc -l)" -ge 37 ] && [ "$(ls $S/wod/part-*.parquet 2>/dev/null | wc -l)" -ge 41 ]; do sleep 30; done
echo "batch complete $(date +%T)"
while pgrep -u $USER -f "sam_detect detect" >/dev/null; do sleep 10; done
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 $DATA_DIR/envs/sam3/bin/python -m jevdrive.sam_detect latency --list $L/p5_prof200.parquet --tag q4d-latency
LAT=$(ls -td $DATA_DIR/runs/fusion_diag/q4/q4d-latency/*/ | head -1)latency.json
cat $LAT
J=$DATA_DIR/envs/jevdrive/bin/python
taskset -c 150-179 $J -m jevdrive.fusion_q8 --latency $LAT
taskset -c 150-179 $J -m jevdrive.fusion_q4 nusc_report --dets $S/nusc
taskset -c 150-179 $J -m jevdrive.fusion_q2b --dets $S/wod
