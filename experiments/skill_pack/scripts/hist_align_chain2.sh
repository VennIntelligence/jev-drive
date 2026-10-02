#!/usr/bin/env bash
# History alignment, phase 2 (plans/2026-10-04-history-align-plan.md, addendum 3): navtrain-calibrated confidence selector.
# Waits for the navtest `straight` rollout, then: navtest straight + sel-straight official scoring (background); navtrain
# rollouts of rot0 / straight -> official navtrain subset PDMS -> ratio x per rule -> pose files sel-<rule>-r<x> ->
# official navhard EPDMS; exact navtest readout of every ratio. STATUS / DONE / ERROR in $DATA_DIR/runs/skill_pack/hist_align.
#   scripts/tmux_run.sh halign2 experiments/skill_pack/scripts/hist_align_chain2.sh [gpu=1]
set -uo pipefail
repo=$(cd "$(dirname "$0")/../../.." && pwd); cd "$repo"
GPU=${1:-1}; PROCS=${PROCS:-3}
R=$DATA_DIR/runs/skill_pack/hist_align; rm -f "$R/ERROR" "$R/DONE2"
E=$DATA_DIR/envs; P=$DATA_DIR/runs/op_lb; CPUS="0-$(($(nproc --all) - 1))"
st() { echo "$(date '+%F %T') [2] $*" | tee -a "$R/STATUS"; }
die() { st "ERROR: $*"; echo "$*" > "$R/ERROR"; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID OPI_ROOT=op_lb
JV=$E/jevdrive/bin/python; NV=$E/navsim2/bin/python
score() {   # official score of one pose file: score <v1|v2> <split> <data> <arm>
  local n=opi_${3}_gimm-cinque_al-${4}__base
  ls "$DATA_DIR"/runs/navsim/eval/${1}_${2}_$n/*/*.csv > /dev/null 2>&1 && return 0
  env ${SUB:-} NAVSIM_THREADS=16 taskset -c "$CPUS" experiments/zeroshot_openloop/archive/navsim_zs_score.sh score "$1" "$2" "$n" \
    "$P/$3/preds/gimm-cinque_al-${4}__base.npz" > "$P/$3/score_$n.log" 2>&1
  st "$1 $2 $4: $(grep -a 'Final' "$P/$3/score_$n.log" | tail -1 | awk '{print $NF}')"
}

st "waiting for the navtest straight rollout"
while [[ ! -f $P/lb_navtest/plans/gimm@cinque_al-straight.npz ]]; do sleep 60; done
$JV experiments/op_openloop/lib/op_interp.py nav-export --data lb_navtest --plans gimm@cinque_al-straight >> "$R/export.log" 2>&1 || die "export navtest straight"
$JV experiments/skill_pack/scripts/hist_align_select.py --rule straight --data lb_navtest >> "$R/select.log" 2>&1 || die "sel-straight navtest"
( score v1 navtest lb_navtest sel-straight; score v1 navtest lb_navtest straight ) &
bg=$!

for r in rot0 straight; do
  if [[ ! -f $P/lb_navtrain/plans/gimm@cinque_al-$r.npz ]]; then
    st "run lb_navtrain $r (GPU $GPU)"
    CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2 $E/openpilot/bin/python scripts/op_lb.py run --data lb_navtrain --frames gimm --model cinque \
      --align "$r" --procs "$PROCS" >> "$R/run_lb_navtrain.log" 2>&1 || die "run navtrain $r"
  fi
  $JV experiments/op_openloop/lib/op_interp.py nav-export --data lb_navtrain --plans "gimm@cinque_al-$r" >> "$R/export.log" 2>&1 || die "export navtrain $r"
  SUB="SUBSET=1 CACHE_NAME=v1_navtrain_oplb TOKENS_FILE=$P/lb_navtrain/tokens.txt" score v1 navtrain lb_navtrain "$r"
done
st "navtrain selector readout"
$NV experiments/skill_pack/scripts/hist_align_report.py select-v1 --data lb_navtrain --arms rot0 straight > "$R/select_navtrain.log" 2>&1 || die "select navtrain"
for r in rot0 straight; do
  x=$($NV -c "
import json; d = json.load(open('experiments/skill_pack/results/history-align/select_navtrain.json'))['$r']
ok = [x for x in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5) if d[f'r{x:g}']['delta'] >= -0.15]
print(f'{max(ok):g}' if ok else '')")
  st "navtrain ratio for $r: ${x:-none}"
  [[ -z $x ]] && continue
  arm=sel-$r; [[ $x != 1 ]] && arm=sel-$r-r$x
  $JV experiments/skill_pack/scripts/hist_align_select.py --rule "$r" --ratio "$x" --data lb_navhard lb_navtest >> "$R/select.log" 2>&1 || die "select $r"
  score v2 navhard_two_stage lb_navhard "$arm"
done
wait $bg
st "navtest selector readout (exact decomposition)"
$NV experiments/skill_pack/scripts/hist_align_report.py select-v1 --data lb_navtest --arms rot0 straight > "$R/select_navtest.log" 2>&1 || die "select navtest"
st "done"; touch "$R/DONE2"
