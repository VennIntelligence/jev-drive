#!/usr/bin/env bash
# NAVSIM raise arm N2 (todos/2026-09-30-navsim-raise.md): 12-config dev grid on N1's held-out logs (+ fold-1 replicate)
# -> refit of the chosen config on all rows -> navtest scored once -> report -> navhard scored once. GPU 6 (small).
#   scripts/tmux_run.sh rn2 env CUDA_VISIBLE_DEVICES=6 CUDA_DEVICE_ORDER=PCI_BUS_ID scripts/navsim_raise_n2.sh <looks>
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd); cd "$repo"
LOOKS=${1:-4}; CPUS=${RAISE_CPUS:-155-183}; SP=${RAISE_SP:-8}
R=$DATA_DIR/runs/skill_pack/raise/n2; mkdir -p "$R"; rm -f "$R/ERROR"
JV=$DATA_DIR/envs/jevdrive/bin/python
st() { echo "$(date '+%F %T') $*" | tee -a "$R/STATUS"; }
die() { st "ERROR: $*"; echo "$*" > "$R/ERROR"; exit 1; }
[[ ${CUDA_VISIBLE_DEVICES:-} == 6 ]] || die "CUDA_VISIBLE_DEVICES must be 6"
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
T() { taskset -c "$CPUS" nice -n 19 "$@"; }
scored() { ls "$DATA_DIR"/runs/navsim/eval/$1/*/*.csv > /dev/null 2>&1; }
[[ -f $R/n2dev.json ]] || { st "dev grid (held-out only)"; T $JV -m jevdrive.navsim_raise n2dev >> "$R/log.txt" 2>&1 || die n2dev; }
[[ -f $R/navhard_n2.npz ]] || { st "final refit"; T $JV -m jevdrive.navsim_raise n2final >> "$R/log.txt" 2>&1 || die n2final; }
scored v1_navtest_sp_n2_navtest || { st "score navtest N2 (the one test run)"
  NAVSIM_THREADS=$SP T scripts/navsim_zs_score.sh score v1 navtest sp_n2_navtest "$R/navtest_n2.npz" > "$R/score_navtest.log" 2>&1 || die "score navtest"; }
st "report"; T $JV -m jevdrive.navsim_raise report --arm n2 --looks "$LOOKS" > "$R/report.md" 2>> "$R/log.txt" || die report
scored v2_navhard_two_stage_sp_n2_navhard || { st "score navhard N2 (once)"
  NAVSIM_THREADS=$SP T scripts/navsim_zs_score.sh score v2 navhard_two_stage sp_n2_navhard "$R/navhard_n2.npz" > "$R/score_navhard.log" 2>&1 || die "score navhard"; }
st "DONE n2"; touch "$R/DONE"
