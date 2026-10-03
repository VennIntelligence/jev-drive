#!/usr/bin/env bash
# OD2 step 2: the WOD column (plans/2026-10-04-od2-prereg.md section 2). Port rollouts of shipped / it_dw3-s0, native and rot0, on
# the 479 WOD-E2E val rater frames, then RFS / ADE with the selector at 0.6 (+ the od2_ratio refit value when it differs).
# Usage (box, tmux): GPU=0 experiments/op_adapt_h/scripts/h_od2_wod.sh
# Files: $DATA_DIR/runs/op_adapt_H/chain/od2_wod/{STATUS,DONE,ERROR,run.log,score.log}; plans $DATA_DIR/runs/op_adapt_H/wod/plans.npz;
# results experiments/op_adapt_h/results/one_driver/wod.json.
set -uo pipefail
repo=$(cd "$(dirname "$0")/../../.." && pwd); cd "$repo"
: "${DATA_DIR:?}" "${GPU:?}"
H=$DATA_DIR/runs/op_adapt_H; C=$H/chain/od2_wod; mkdir -p "$C"; rm -f "$C/ERROR" "$C/DONE"
PY=$DATA_DIR/envs/op-train/bin/python; CPUS=${CPUS:-0-39}
st() { echo "$(date '+%F %T') $*" | tee -a "$C/STATUS"; }
die() { st "ERROR: $*"; echo "$*" > "$C/ERROR"; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1
free=$(df -BG --output=avail "$DATA_DIR" | tail -1 | tr -dc 0-9); (( free >= 150 )) || die "data disk free ${free} GB < 150"
if [[ ! -f $H/wod/plans.npz ]]; then
  st "run (GPU $GPU, CPUs $CPUS)"
  CUDA_VISIBLE_DEVICES=$GPU taskset -c "$CPUS" $PY experiments/op_adapt_h/scripts/h_wod.py run >> "$C/run.log" 2>&1 || die "run"
fi
R=(0.6); x=$(cat "$H/chain/od2_ratio/RATIO" 2>/dev/null || true)
[[ -n $x && $x != none && $x != 0.6 ]] && R+=("$x")
st "score ratios ${R[*]}"
CUDA_VISIBLE_DEVICES= taskset -c "$CPUS" $PY experiments/op_adapt_h/scripts/h_wod.py score --ratios "${R[@]}" > "$C/score.log" 2>&1 || die "score"
st "done"; touch "$C/DONE"
