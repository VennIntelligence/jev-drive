#!/usr/bin/env bash
# OD2 step 1: refit the selector ratio for the adapted model on navtrain only (decision 94's procedure, history-align plan
# addendum 3): op_lb rollouts of the adapted ONNX on lb_navtrain (native and --align rot0), official v1 navtrain subset PDMS
# of both, x = the largest ratio in {1.0 .. 0.5} with navtrain delta >= -0.15. If x != 0.6: selector pose files at x on
# navhard / navtest (the rollouts exist from h_onedriver_nav.sh), official scoring, paired reports. The 0.6 row stays.
# Usage (box, tmux): GPU=0 PROCS=4 experiments/op_adapt_h/scripts/h_od2_ratio.sh [tag=it_dw3-s0]
# Files: $DATA_DIR/runs/op_adapt_H/chain/od2_ratio/{STATUS,DONE,ERROR}; results in experiments/op_adapt_h/results/one_driver/.
set -uo pipefail
repo=$(cd "$(dirname "$0")/../../.." && pwd); cd "$repo"
: "${DATA_DIR:?}" "${GPU:?}"
T=${1:-it_dw3-s0}; O=O$T; PROCS=${PROCS:-4}
H=$DATA_DIR/runs/op_adapt_H; C=$H/chain/od2_ratio; mkdir -p "$C"; rm -f "$C/ERROR" "$C/DONE"
E=$DATA_DIR/envs; P=$DATA_DIR/runs/op_lb; CPUS=${CPUS:-"0-$(($(nproc --all) - 1))"}
JV=$E/jevdrive/bin/python; NV=$E/navsim2/bin/python
st() { echo "$(date '+%F %T') $*" | tee -a "$C/STATUS"; }
die() { st "ERROR: $*"; echo "$*" > "$C/ERROR"; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=2 NAVSIM_THREADS=${NAVSIM_THREADS:-16} OPI_ROOT=op_lb
ONNX=$H/onnx/$T.onnx; [[ -f $ONNX ]] || die "missing $ONNX"
OUTD=$repo/experiments/op_adapt_h/results/one_driver; mkdir -p "$OUTD"; export REPORT_OUT=$OUTD
B=gimm-cinque_$O
score() {   # official score of one pose file: score <v1|v2> <split> <data> <pred stem>
  local n=opi_${3}_${4}__base
  ls "$DATA_DIR"/runs/navsim/eval/${1}_${2}_$n/*/*.csv > /dev/null 2>&1 && return 0
  env ${SUB:-} taskset -c "$CPUS" experiments/zeroshot_openloop/archive/navsim_zs_score.sh score "$1" "$2" "$n" \
    "$P/$3/preds/${4}__base.npz" > "$P/$3/score_$n.log" 2>&1 || die "score $1 $2 $4"
  st "$1 $2 $4: $(grep -a 'Final' "$P/$3/score_$n.log" | tail -1 | awk '{print $NF}')"
}

for al in none rot0; do
  stem=gimm@cinque_$O$([[ $al == none ]] || echo _al-$al)
  if [[ ! -f $P/lb_navtrain/plans/$stem.npz ]]; then
    st "run lb_navtrain $al (GPU $GPU)"
    CUDA_VISIBLE_DEVICES=$GPU $E/openpilot/bin/python scripts/op_lb.py run --data lb_navtrain --frames gimm --model cinque --onnx "$ONNX" \
      --tag "$O" --align "$al" --procs "$PROCS" >> "$C/run_lb_navtrain.log" 2>&1 || die "run navtrain $al"
  fi
  $JV experiments/op_openloop/lib/op_interp.py nav-export --data lb_navtrain --plans "$stem" >> "$C/export.log" 2>&1 || die "export navtrain $al"
  SUB="SUBSET=1 CACHE_NAME=v1_navtrain_oplb TOKENS_FILE=$P/lb_navtrain/tokens.txt" score v1 navtrain lb_navtrain "$B$([[ $al == none ]] || echo _al-$al)"
done
st "navtrain selector readout"
$NV experiments/skill_pack/scripts/hist_align_report.py select-v1 --data lb_navtrain --base-stem "$B" --arms rot0 --tag "_$T" > "$C/select_navtrain.log" 2>&1 || die "select navtrain"
$NV experiments/skill_pack/scripts/hist_align_report.py select-v1 --data lb_navtest --base-stem "$B" --arms rot0 --tag "_$T" > "$C/select_navtest.log" 2>&1 || die "select navtest"
x=$($NV -c "
import json; d = json.load(open('$OUTD/select_navtrain_$T.json'))['rot0']
ok = [x for x in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5) if d[f'r{x:g}']['delta'] >= -0.15]
print(f'{max(ok):g}' if ok else '')")
st "navtrain ratio for $T rot0: ${x:-none}"; echo "${x:-none}" > "$C/RATIO"
if [[ -n $x && $x != 0.6 ]]; then
  arm=sel-rot0; [[ $x != 1 ]] && arm=sel-rot0-r$x
  $JV experiments/skill_pack/scripts/hist_align_select.py --rule rot0 --tag "$O" --ratio "$x" --data lb_navhard lb_navtest >> "$C/select.log" 2>&1 || die "select $x"
  score v2 navhard_two_stage lb_navhard "${B}_al-$arm"
  score v1 navtest lb_navtest "${B}_al-$arm"
  R=experiments/skill_pack/scripts/hist_align_report.py
  st "paired reports ($arm)"
  $NV $R navtest --arms "${B}_al-$arm" --tag "_${arm}_vs_shipped" > "$C/rep_navtest1.log" 2>&1 || die "navtest report 1"
  $NV $R navtest --base-stem "$B" --arms "$arm" --tag "_${arm}_vs_native" > "$C/rep_navtest2.log" 2>&1 || die "navtest report 2"
  $NV $R navhard --arms "${B}_al-$arm" --tag "_${arm}_vs_shipped" --procs 24 > "$C/rep_navhard1.log" 2>&1 || die "navhard report 1"
  $NV $R navhard --base-stem "$B" --arms "$arm" --tag "_${arm}_vs_native" --procs 24 > "$C/rep_navhard2.log" 2>&1 || die "navhard report 2"
fi
st "done"; touch "$C/DONE"
