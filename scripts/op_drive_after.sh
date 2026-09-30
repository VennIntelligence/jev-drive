#!/usr/bin/env bash
# op-drive follow-up chain on GPU 1 (one card, 6 CARLA; todos/2026-09-29-op-drive.md): waits for the seed-0 dev chain, pulls the code
# (nothing of ours is running at that point), finishes the seed-1 dev chain (dbaseslow + pacing), re-checks seed-0 pacing with the
# 3-iteration loop, then runs R3a (dtl) on dev, seeds 0 and 1. Resumable: every step skips what has a DONE file.
# Resume after a box restart:  cd ~/data/jev-drive && GPU=1 IDX0=300 WORKERS=6 CUDA_VISIBLE_DEVICES=1 scripts/op_drive_after.sh
# Hand-offs in $OP_ARB_DIR: DONE-after, ERROR.
set -uo pipefail
: "${DATA_DIR:?}" "${GPU:?}" "${IDX0:?}"
cd "$(dirname "$0")/.."
export OP_ARB_DIR=${OP_ARB_DIR:-$DATA_DIR/runs/op_drive_g$GPU} OP_ARB_ARMS=${OP_ARB_ARMS:-$DATA_DIR/runs/op_drive/arms}
export WORKERS=${WORKERS:-6} CPUS=${CPUS:-16-47}
until [[ -e $OP_ARB_DIR/DONE-dev || -e $OP_ARB_DIR/ERROR ]]; do sleep 30; done
[[ -e $OP_ARB_DIR/ERROR ]] && exit 1
git pull -q || { echo "pull failed $(date)" > "$OP_ARB_DIR/ERROR"; exit 1; }
for s in 1 0; do
    rm -f "$OP_ARB_DIR/DONE-dev"; SEED=$s scripts/op_drive_dev.sh || exit 1
done
export SEEDS="0 1" ARMS=dtl LAT_EXEC=curv
scripts/op_arb.sh set 2 f || { echo "dtl failed $(date)" > "$OP_ARB_DIR/ERROR"; exit 1; }
date > "$OP_ARB_DIR/DONE-after"
