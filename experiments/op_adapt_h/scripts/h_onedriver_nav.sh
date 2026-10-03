#!/usr/bin/env bash
# One-driver table, navtest + navhard cells of it_dw3-s0 + selector (decision 94's sel-rot0-r0.6 on the adapted model):
# op_lb rollouts of the adapted serving ONNX (native and --align rot0) on both splits, selector, official scoring, paired reports.
# Usage (box, tmux): GPU=0 PROCS=4 experiments/op_adapt_h/scripts/h_onedriver_nav.sh [tag=it_dw3-s0]
# Files: $DATA_DIR/runs/op_adapt_H/chain/onedriver_nav/{STATUS,DONE,ERROR}; results copied to experiments/op_adapt_h/results/one_driver/.
set -uo pipefail
repo=$(cd "$(dirname "$0")/../../.." && pwd); cd "$repo"
: "${DATA_DIR:?}" "${GPU:?}"
T=${1:-it_dw3-s0}; O=O$T; PROCS=${PROCS:-4}
H=$DATA_DIR/runs/op_adapt_H; C=$H/chain/onedriver_nav; mkdir -p "$C"; rm -f "$C/ERROR" "$C/DONE"
E=$DATA_DIR/envs; P=$DATA_DIR/runs/op_lb
st() { echo "$(date '+%F %T') $*" | tee -a "$C/STATUS"; }
die() { st "ERROR: $*"; echo "$*" > "$C/ERROR"; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=2 NAVSIM_THREADS=16 OPI_ROOT=op_lb
ONNX=$H/onnx/$T.onnx; [[ -f $ONNX ]] || die "missing $ONNX"
OUTD=$repo/experiments/op_adapt_h/results/one_driver; mkdir -p "$OUTD"
for data in lb_navhard lb_navtest; do
  for al in none rot0; do
    stem=gimm@cinque_$O$([[ $al == none ]] || echo _al-$al)
    if [[ ! -f $P/$data/plans/$stem.npz ]]; then
      st "run $data $al"
      CUDA_VISIBLE_DEVICES=$GPU $E/openpilot/bin/python scripts/op_lb.py run --data "$data" --frames gimm --model cinque --onnx "$ONNX" --tag "$O" \
        --align "$al" --procs "$PROCS" >> "$H/chain/onedriver_nav/run_$data.log" 2>&1 || die "run $data $al"
    fi
    $E/jevdrive/bin/python experiments/op_openloop/lib/op_interp.py nav-export --data "$data" --plans "$stem" >> "$C/export.log" 2>&1 || die "export $data $al"
  done
  $E/openpilot/bin/python experiments/skill_pack/scripts/hist_align_select.py --rule rot0 --tag "$O" --ratio 0.6 --data "$data" >> "$C/select.log" 2>&1 || die "select $data"
done
st "official scoring"
experiments/op_openloop/archive/op_interp_score.sh "0-$(($(nproc --all) - 1))" lb_navhard v2 navhard_two_stage > "$C/score_navhard.log" 2>&1 || die "score navhard"
experiments/op_openloop/archive/op_interp_score.sh "0-$(($(nproc --all) - 1))" lb_navtest v1 navtest > "$C/score_navtest.log" 2>&1 || die "score navtest"
export REPORT_OUT=$OUTD
R=experiments/skill_pack/scripts/hist_align_report.py
SEL=gimm-cinque_${O}_al-sel-rot0-r0.6
st "navtest paired"
$E/navsim2/bin/python $R navtest --arms gimm-cinque_$O $SEL --tag _vs_shipped > "$C/rep_navtest1.log" 2>&1 || die "navtest report 1"
$E/navsim2/bin/python $R navtest --base-stem gimm-cinque_$O --arms $SEL --tag _sel_vs_native > "$C/rep_navtest2.log" 2>&1 || die "navtest report 2"
st "navhard paired"
$E/navsim2/bin/python $R navhard --arms gimm-cinque_$O $SEL --tag _vs_shipped --procs 24 > "$C/rep_navhard1.log" 2>&1 || die "navhard report 1"
$E/navsim2/bin/python $R navhard --base-stem gimm-cinque_$O --arms $SEL --tag _sel_vs_native --procs 24 > "$C/rep_navhard2.log" 2>&1 || die "navhard report 2"
st "done"; touch "$C/DONE"
