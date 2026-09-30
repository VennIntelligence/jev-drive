#!/usr/bin/env bash
# op-drive resume-policy tuning (todos/2026-09-29-op-drive.md, "registration R"): drive on the 6 tuning routes with resume
# variants R0 (timer), R1 (nored) and R2 (no resume), one seed on one card. One self-advancing chain; hand-offs in $OP_ARB_DIR:
# STATUS, DONE-tune, ERROR. Usage: GPU=1 IDX0=300 SEED=0 WORKERS=4 CPUS=16-47 scripts/op_drive_tune.sh
set -uo pipefail
: "${DATA_DIR:?}" "${GPU:?}" "${IDX0:?}" "${SEED:?}"
cd "$(dirname "$0")/.."
export OP_ARB_ARMS=${OP_ARB_ARMS:-$DATA_DIR/runs/op_drive/arms} OP_ARB_DIR=${OP_ARB_DIR:-$DATA_DIR/runs/op_drive_g$GPU}
export GPU IDX0 WORKERS=${WORKERS:-4} CPUS=${CPUS:-16-47} SEEDS=$SEED LAT_EXEC=curv ARMS=drive
mkdir -p "$OP_ARB_DIR"
rm -f "$OP_ARB_DIR/ERROR"
run() { scripts/op_arb.sh set 1 "$1" || { echo "set $1 failed rc=$? $(date)" > "$OP_ARB_DIR/ERROR"; exit 1; }; }
DRIVE_ARGS= RESUME_S=5 run r0
DRIVE_ARGS='"resume": "nored"' RESUME_S=5 run r1
DRIVE_ARGS= RESUME_S=1e9 run r2
date > "$OP_ARB_DIR/DONE-tune"
