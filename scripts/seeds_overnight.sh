#!/usr/bin/env bash
# Overnight queue item 2 ([SEEDS], todos/2026-09-26-overnight-queue.md): seeds of the M-C, NAVSIM, WOD and Q9b heads.
#   scripts/seeds_overnight.sh check    seed 0 of M-C (BA), NAVSIM heads and WOD heads, to compare with the stored runs
#   scripts/seeds_overnight.sh seeds    seeds 1, 2 of everything (GPU), then devkit scoring of the NAVSIM seeds (CPU)
# GPU from $GPU (default 0); run inside tmux.
set -uo pipefail
cd ~/data/jev-drive
PY=.venv/bin/python
G=${GPU:-0}
run() { env CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 nice -n 10 "$@"; }
mc()   { run env P5_SET=carla_p5v1_$1 $PY -m jevdrive.reactivity_mc --op-sub op_streams_vis --fold-seed $2; }
nav()  { run $PY -m jevdrive.navsim_heads fit --seed $1; }
wod()  { run $PY -m jevdrive.drive_backbones --steps heads_train --models cinque,lebowski --seed $1; }
q9b()  { run env P5_SET=carla_p5v1_$1 $PY -m jevdrive.fusion_q9b fit --op-sub op_streams_vis --tag v1-$1 \
           --out ../elicitation/seeds/q9b-v1-$1-hard$2 --hard-seed $2; }
case $1 in
  check) mc ba 0 && nav 0 && wod 0 ;;
  seeds)
    for s in 1 2; do mc ba $s; mc pdm $s; done
    mc pdm 0
    for s in 1 2; do nav $s; done
    for s in 1 2; do q9b ba $s; q9b pdm $s; done
    for s in 1 2; do wod $s; done
    echo "GPU part done $(date +%H:%M)"
    S() { ls $DATA_DIR/runs/navsim/eval/${1}_${2}_$3/*/*.csv >/dev/null 2>&1 && return 0
          NAVSIM_THREADS=12 nice -n 10 scripts/navsim_zs_score.sh score "$@" > /tmp/seeds-score-$1-$2-$3.log 2>&1; }
    arm_score() {  # arm_score <run dir> <seed> <arm>
      S v1 navtest heads_s$2_$3 $1/navtest_$3.npz; S v2 navtest heads_s$2_$3 $1/navtest_$3.npz
      S v2 navhard_two_stage heads_s$2_$3 $1/navhard_two_stage_$3.npz; }
    for s in 1 2; do
      r=$(ls -td $DATA_DIR/runs/navsim_zs/heads-s$s/*/ | head -1)
      for m in cinque lebowski; do
        arm_score $r $s ridge_late_${m}_temporal & arm_score $r $s cls_late_${m}_temporal; wait
      done
      arm_score $r $s ridge_ego & arm_score $r $s cls_ego_K1024; wait      # the paired deltas' ego rows
    done
    r=$(ls -td $DATA_DIR/runs/navsim_zs/heads-s0/*/ | head -1)      # s0': the cls heads' run-to-run noise (log 01:31)
    for a in cls_late_cinque_temporal cls_late_lebowski_temporal cls_ego_K1024; do
      S v1 navtest heads_s0r_$a $r/navtest_$a.npz & S v2 navtest heads_s0r_$a $r/navtest_$a.npz; wait
    done
    echo "scoring done $(date +%H:%M)" ;;
esac
