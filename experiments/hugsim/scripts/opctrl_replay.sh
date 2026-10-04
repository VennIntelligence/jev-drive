#!/usr/bin/env bash
# Offline part of plans/2026-10-05-op-control-stack-prereg.md: CPU replay of the 64 exam-base runs (opctrl_replay.py), N parts in parallel.
# Usage (box, tmux): [PARTS=8] [THREADS=6] [STEPS=40] experiments/hugsim/scripts/opctrl_replay.sh [out_dir]   -> <out>/DONE or ERROR
set -uo pipefail
: "${DATA_DIR:?}"
cd "$(dirname "$0")/../../.."
OUT=${1:-$DATA_DIR/runs/opctrl/replay}
N=${PARTS:-8}
mkdir -p "$OUT"; rm -f "$OUT/DONE" "$OUT/ERROR"
pids=()
for i in $(seq 0 $((N - 1))); do
    "$DATA_DIR/envs/openpilot/bin/python" -u experiments/hugsim/scripts/opctrl_replay.py "$DATA_DIR/runs/hugsim-exam/scored-op/results.csv" "$OUT" \
        --part "$i/$N" --steps "${STEPS:-40}" --threads "${THREADS:-6}" > "$OUT/part$i.log" 2>&1 &
    pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
n=$(ls "$OUT"/*.json 2>/dev/null | wc -l)
if (( rc == 0 && n >= 64 )); then touch "$OUT/DONE"; else echo "rc $rc, $n of 64" > "$OUT/ERROR"; fi
