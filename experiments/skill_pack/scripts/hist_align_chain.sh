#!/usr/bin/env bash
# History-alignment rule, full navhard + navtest (experiments/skill_pack/plans/2026-10-04-history-align-plan.md).
# Resumable; STATUS / DONE / ERROR in $DATA_DIR/runs/skill_pack/hist_align.
#   scripts/tmux_run.sh halign experiments/skill_pack/scripts/hist_align_chain.sh [gpu=1] [rules="rot0 straight straight_keys"]
# GPU: Cinque runs ($PROCS shards, ~1.5 GB each) on one card; CPU: official scorers (16 threads) then the paired harness (24 procs).
set -uo pipefail
repo=$(cd "$(dirname "$0")/../../.." && pwd); cd "$repo"
GPU=${1:-1}; RULES=${2:-rot0 straight straight_keys}; PROCS=${PROCS:-3}
R=$DATA_DIR/runs/skill_pack/hist_align; mkdir -p "$R"; rm -f "$R/ERROR" "$R/DONE"
E=$DATA_DIR/envs
st() { echo "$(date '+%F %T') $*" | tee -a "$R/STATUS"; }
die() { st "ERROR: $*"; echo "$*" > "$R/ERROR"; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=2 NAVSIM_THREADS=16 OPI_ROOT=op_lb   # op_lb run dirs for export and scoring
P=$DATA_DIR/runs/op_lb

run_split() {   # model runs + pose export for one split
  local data=$1
  for r in $RULES; do
    local stem=gimm@cinque_al-$r
    if [[ ! -f $P/$data/plans/$stem.npz ]]; then
      st "run $data $r (GPU $GPU)"
      CUDA_VISIBLE_DEVICES=$GPU $E/openpilot/bin/python scripts/op_lb.py run --data "$data" --frames gimm --model cinque --align "$r" --procs "$PROCS" \
        >> "$R/run_$data.log" 2>&1 || die "run $data $r"
    fi
    OPI_ROOT=op_lb $E/jevdrive/bin/python experiments/op_openloop/lib/op_interp.py nav-export --data "$data" --plans "$stem" >> "$R/export.log" 2>&1 \
      || die "export $data $r"
  done
}

run_split lb_navhard
st "official navhard scoring (background)"
experiments/op_openloop/archive/op_interp_score.sh "" lb_navhard v2 navhard_two_stage > "$R/score_navhard.log" 2>&1 &
sp=$!
run_split lb_navtest
wait $sp || die "navhard scoring"
st "official navtest scoring"
experiments/op_openloop/archive/op_interp_score.sh "" lb_navtest v1 navtest > "$R/score_navtest.log" 2>&1 || die "navtest scoring"
st "navtest report"
$E/navsim2/bin/python experiments/skill_pack/scripts/hist_align_report.py navtest --arms $RULES > "$R/report_navtest.log" 2>&1 || die "navtest report"
st "navhard paired harness"
$E/navsim2/bin/python experiments/skill_pack/scripts/hist_align_report.py navhard --arms $RULES --procs 24 > "$R/report_navhard.log" 2>&1 || die "navhard report"
st "done"; touch "$R/DONE"
