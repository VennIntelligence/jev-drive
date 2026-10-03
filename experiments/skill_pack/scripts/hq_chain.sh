#!/usr/bin/env bash
# Lane C (history quality), scored stage (plans/2026-10-04-history-quality-prereg.md): the history ladder hold / warp / GIMM /
# real on the navtest subset lb_hq_navtest, rotL / rotR yaw probes, the navhard stage-1 subset, official v1 PDMS on the
# subset, readouts (hq_report.py). Waits for hq_fetch.sh. Resumes from disk. STATUS / DONE / ERROR in $DATA_DIR/runs/op_lb/hq/chain.
#   GPU=2 scripts/tmux_run.sh hqchain experiments/skill_pack/scripts/hq_chain.sh
set -uo pipefail
repo=$(cd "$(dirname "$0")/../../.." && pwd); cd "$repo"
R=$DATA_DIR/runs/op_lb/hq/chain; mkdir -p "$R"; rm -f "$R/ERROR" "$R/DONE"
GPU=${GPU:-2}; PROCS=${PROCS:-4}; ST=${ST:-24}
E=$DATA_DIR/envs; P=$DATA_DIR/runs/op_lb; H=experiments/skill_pack/scripts/hq_real.py
JV=$E/jevdrive/bin/python; NV=$E/navsim2/bin/python; OPY=$E/openpilot/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID OPI_ROOT=op_lb
st() { echo "$(date '+%F %T') $*" | tee -a "$R/STATUS"; }
die() { st "ERROR: $*"; echo "$*" > "$R/ERROR"; exit 1; }
until [[ -f $P/hq/DONE ]]; do [[ -f $P/hq/ERROR ]] && die "fetch stage failed"; sleep 30; done
[[ -f $P/lb_hq_navtest/hold.npy ]] || $JV $H hold --data lb_hq_navtest >> "$R/hold.log" 2>&1 || die hold

roll() {   # roll <data> <frames> [align]: stem <frames>@cinque[_al-<align>]
  local data=$1 fr=$2 al=${3:-none} s
  s="$fr@cinque$([[ $al == none ]] || echo "_al-$al")"
  if [[ ! -f $P/$data/plans/$s.npz ]]; then
    st "roll $data $s"
    CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 $OPY scripts/op_lb.py run --data "$data" --frames "$fr" --model cinque --align "$al" \
      --procs "$PROCS" $([[ $fr == vh187 ]] && echo "--vcam 0 --vcam-workers 6") >> "$R/run_$data.log" 2>&1 || die "roll $data $s"
  fi
}
for d in lb_hq_navtest lb_hq_navhard1; do
  for fr in real gimm vh187 $([[ $d == lb_hq_navtest ]] && echo hold); do roll $d $fr; done
  for fr in real gimm vh187; do for al in rotL rotR; do roll $d $fr $al; done; done
done
ARMS=(hold@cinque vh187@cinque gimm@cinque real@cinque)
$JV experiments/op_openloop/lib/op_interp.py nav-export --data lb_hq_navtest --plans "${ARMS[@]}" >> "$R/export.log" 2>&1 || die export
scored() { ls "$DATA_DIR"/runs/navsim/eval/$1/*/*.csv > /dev/null 2>&1; }
for s in "${ARMS[@]}"; do
  n=opi_lb_hq_navtest_${s/@/-}__base
  scored "v1_navtest_$n" && continue
  st "score $s"
  TOKENS_FILE=$P/lb_hq_navtest/tokens.txt NAVSIM_THREADS=$ST experiments/zeroshot_openloop/archive/navsim_zs_score.sh score v1 navtest "$n" \
    "$P/lb_hq_navtest/preds/${s/@/-}__base.npz" > "$R/score_$s.log" 2>&1 || die "score $s"
  st "scored $s: $(grep -a 'Final' "$R/score_$s.log" | tail -1 | awk '{print $NF}')"
done
st "report"; $NV experiments/skill_pack/scripts/hq_report.py > "$R/report.log" 2>&1 || die report
st done; touch "$R/DONE"
