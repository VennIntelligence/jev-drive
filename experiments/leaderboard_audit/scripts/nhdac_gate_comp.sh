#!/usr/bin/env bash
# Compensate the best arm (it_dw3 + selector) on navhard and navtest, both modes, all alphas (gated compensation follow-up).
# Output: $DATA_DIR/runs/leaderboard_audit/navhard_dac/gated/{lb_navhard,lb_navtest}/best_{full,path}_a*.npz + diag pkls.
set -uo pipefail
repo=$(cd "$(dirname "$0")/../../.." && pwd); cd "$repo"
O=$DATA_DIR/runs/leaderboard_audit/navhard_dac/gated; mkdir -p "$O"; rm -f "$O/ERROR_comp" "$O/DONE_comp"
export PYTHONPATH=$DATA_DIR/third_party/navsim:$DATA_DIR/third_party/nuplan-devkit OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
B=gimm-cinque_Oit_dw3-s0_al-sel-rot0-r0.6__base.npz
for d in lb_navhard lb_navtest; do
  [[ -f $O/diag_${d}_full-path.pkl ]] && continue
  echo "$(date +%T) compensate $d" >> "$O/STATUS"
  $DATA_DIR/envs/navsim2/bin/python experiments/skill_pack/scripts/trk_precomp.py make --data $d --alphas 0.25 0.5 0.75 1 \
    --src best=$DATA_DIR/runs/op_lb/$d/preds/$B --modes full path --procs 100 --out "$O" >> "$O/comp_$d.log" 2>&1 || { echo "make $d" > "$O/ERROR_comp"; exit 1; }
done
echo "$(date +%T) done" >> "$O/STATUS"; touch "$O/DONE_comp"
