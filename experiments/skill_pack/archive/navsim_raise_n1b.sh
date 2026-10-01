#!/usr/bin/env bash
# NAVSIM raise N1b chain (fc65452:todos/2026-09-30-navsim-raise.md): navhard Cinque features -> N1 refit (repro check) ->
# N1b fit (widened lambda grid, held-out only) -> navtest scored once for N1b -> navhard scored once each for N1, N1b.
# GPU 6 only (small), resumable, writes STATUS / DONE / ERROR in $DATA_DIR/runs/skill_pack/raise.
#   scripts/tmux_run.sh raise env CUDA_VISIBLE_DEVICES=6 CUDA_DEVICE_ORDER=PCI_BUS_ID experiments/skill_pack/archive/navsim_raise_n1b.sh
set -uo pipefail
repo=$(cd "$(dirname "$0")/../../.." && pwd); cd "$repo"
CPUS=${RAISE_CPUS:-155-183}; PROCS=${RAISE_PROCS:-6}; SP=${RAISE_SP:-8}
R=$DATA_DIR/runs/skill_pack/raise; mkdir -p "$R"; rm -f "$R/ERROR"
E=$DATA_DIR/envs; OP=$E/openpilot/bin/python; JV=$E/jevdrive/bin/python
st() { echo "$(date '+%F %T') $*" | tee -a "$R/STATUS"; }
die() { st "ERROR: $*"; echo "$*" > "$R/ERROR"; exit 1; }
[[ ${CUDA_VISIBLE_DEVICES:-} == 6 ]] || die "CUDA_VISIBLE_DEVICES must be 6 (got '${CUDA_VISIBLE_DEVICES:-}')"
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
T() { taskset -c "$CPUS" nice -n 19 "$@"; }
F=$DATA_DIR/runs/skill_pack/n1/feat
scored() { ls "$DATA_DIR"/runs/navsim/eval/$1/*/*.csv > /dev/null 2>&1; }

if [[ ! -f $F/lb_navhard_n5912.npz ]]; then
  st "navhard features (Cinque on the lane's GIMM cache), $PROCS shards"
  T $OP experiments/skill_pack/archive/n1_extract.py extract --data lb_navhard --n 5912 --procs "$PROCS" >> "$R/log.txt" 2>&1 || die "extract navhard"
  T $OP experiments/skill_pack/archive/n1_extract.py merge --data lb_navhard --n 5912 >> "$R/log.txt" 2>&1 || die "merge navhard"
fi
for arm in n1 n1b; do
  [[ -f $R/$arm/navhard_$arm.npz ]] || { st "fit $arm --final"; T $JV -m experiments.skill_pack.archive.navsim_raise fit --arm $arm --final >> "$R/log.txt" 2>&1 || die "fit $arm"; }
done
st "N1 repro check"; T $JV -m experiments.skill_pack.archive.navsim_raise repro >> "$R/log.txt" 2>&1 || die "N1 repro check failed"

scored v1_navtest_sp_n1b_navtest || { st "score navtest N1b (the one test run)"
  NAVSIM_THREADS=$SP T experiments/zeroshot_openloop/archive/navsim_zs_score.sh score v1 navtest sp_n1b_navtest "$R/n1b/navtest_n1b.npz" > "$R/score_navtest_n1b.log" 2>&1 || die "score navtest"; }
st "report"; T $JV -m experiments.skill_pack.archive.navsim_raise report --arm n1b > "$R/n1b/report.md" 2>> "$R/log.txt" || die report
for arm in n1 n1b; do
  scored v2_navhard_two_stage_sp_${arm}_navhard || { st "score navhard $arm (once)"
    NAVSIM_THREADS=$SP T experiments/zeroshot_openloop/archive/navsim_zs_score.sh score v2 navhard_two_stage sp_${arm}_navhard "$R/$arm/navhard_$arm.npz" > "$R/score_navhard_$arm.log" 2>&1 || die "score navhard $arm"; }
done
st "DONE n1b"; touch "$R/DONE_n1b"
