#!/usr/bin/env bash
# Real-data transfer G0, NAVSIM column (todos/2026-09-26-real-data-transfer.md, deviation [G0] 08:52): prior + student
# Delta on navtest -> official devkit PDMS (v1.1) and EPDMS (main @ 0a380a9) -> paired table.
# Launch on the box in tmux: scripts/tmux_run.sh g0-navsim scripts/real_g0_navsim.sh
# Re-running skips scored names. JOBS devkit runs at a time, NAVSIM_THREADS ray workers each.
set -uo pipefail
cd ~/data/jev-drive
PY=$DATA_DIR/envs/jevdrive/bin/python
JOBS=${JOBS:-6}
run=$(ls -td $DATA_DIR/runs/real-data-transfer/g0-navsim/*/ 2>/dev/null | head -1)
if [[ -z $run || ! -f $run/navsim_activation.csv ]]; then
  OMP_NUM_THREADS=12 $PY -m jevdrive.real_g0 navsim || exit 1
  run=$(ls -td $DATA_DIR/runs/real-data-transfer/g0-navsim/*/ | head -1)
fi
S() {  # S v1|v2 name npz: score unless a csv for that name exists
  ls $DATA_DIR/runs/navsim/eval/${1}_navtest_$2/*/*.csv >/dev/null 2>&1 && return 0
  NAVSIM_THREADS=${NAVSIM_THREADS:-12} nice -n 10 scripts/navsim_zs_score.sh score "$1" navtest "$2" "$3" > $run/score_${1}_$2.log 2>&1
}
for m in cinque lebowski; do
  for arm in A B; do
    for s in 0 1 2; do
      for v in v1 v2; do
        while (( $(jobs -rp | wc -l) >= JOBS )); do sleep 5; done
        S $v g0_${arm}_s${s}_${m}_plus_student $run/navtest_g0_${arm}_s${s}_${m}.npz &
      done
    done
  done
done
wait
touch $run/scored.done
$PY -m jevdrive.real_g0 navsim-table --runs $run
echo "G0 NAVSIM scored -> $run"
