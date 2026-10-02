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

combined() {   # official two-stage EPDMS of one navhard pose file (empty when not scored)
  local f; f=$(ls "$DATA_DIR"/runs/navsim/eval/v2_navhard_two_stage_opi_lb_navhard_gimm-cinque$1__base/*/*.csv 2>/dev/null | tail -1)
  [[ -n $f ]] && grep -a "extended_pdm_score_combined" "$f" | awk -F, '{print $NF}'
}

RULES_ALL=$RULES
run_split lb_navhard
st "official navhard scoring"
experiments/op_openloop/archive/op_interp_score.sh "0-$(($(nproc --all) - 1))" lb_navhard v2 navhard_two_stage > "$R/score_navhard.log" 2>&1 || die "navhard scoring"
b=$(combined "")
RULES=""
for r in $RULES_ALL; do      # pre-registration: navtest only for rules not rejected on navhard (delta <= 0)
  c=$(combined "_al-$r"); st "navhard official combined: base $b, $r $c"
  [[ -n $c ]] && awk -v c="$c" -v b="$b" 'BEGIN{exit !(c > b)}' && RULES="$RULES $r"
done
st "navtest rules:${RULES:- none}"
[[ -n $RULES ]] && run_split lb_navtest
st "official navtest scoring"
experiments/op_openloop/archive/op_interp_score.sh "0-$(($(nproc --all) - 1))" lb_navtest v1 navtest > "$R/score_navtest.log" 2>&1 || die "navtest scoring"
st "navtest report"
$E/navsim2/bin/python experiments/skill_pack/scripts/hist_align_report.py navtest --arms $RULES_ALL > "$R/report_navtest.log" 2>&1 || die "navtest report"
st "navhard paired harness"
$E/navsim2/bin/python experiments/skill_pack/scripts/hist_align_report.py navhard --arms $RULES_ALL --procs 24 > "$R/report_navhard.log" 2>&1 || die "navhard report"
st "done"; touch "$R/DONE"
