#!/usr/bin/env bash
# Lane B chain (night 2026-10-03): shards on one card, at most 4 openpilot processes at once (each holds ~27 GB RSS;
# eight of them pushed the box's 276 GiB cgroup over), then the report. Each of the 4 slots runs its shards in order.
# Resumable: a finished shard (raw npz) is skipped, an unfinished one resumes from its .part.npz.
# STATUS / DONE / ERROR in $DATA_DIR/runs/op_common_cause/chain/.
#   scripts/tmux_run.sh ccB experiments/op_common_cause/scripts/cc_chain.sh [gpu]
set -uo pipefail
GPU=${1:-1}
cd "$(dirname "$0")/../../.."
L=$DATA_DIR/runs/op_common_cause/chain; mkdir -p "$L"; rm -f "$L/DONE" "$L/ERROR"
PY=$DATA_DIR/envs/openpilot/bin/python
RAW=$DATA_DIR/runs/op_common_cause/raw
st() { echo "$(date '+%F %T') ccB: $*" > "$L/STATUS"; echo "$(date '+%T') $*"; }
SLOTS=("nav:0 wod:0" "nav:1 wod:1" "carla:0 wodt:0" "carla:1 wodt:1")
slot() {
  for job in $1; do
    d=${job%:*}; i=${job#*:}
    [[ -f $RAW/$d/$d-${i}of2.npz ]] && continue
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. $PY experiments/op_common_cause/scripts/cc_run.py --domain "$d" --shard "$i/2" \
      --workers 4 >> "$L/$d-$i.log" 2>&1 || return 1
  done
}
pids=()
for s in "${SLOTS[@]}"; do slot "$s" & pids+=($!); done
st "running 4 slots on GPU $GPU: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
for d in nav wod carla wodt; do for i in 0 1; do
  [[ -f $RAW/$d/$d-${i}of2.npz ]] || { fail=1; echo "missing $d $i/2"; }
done; done
if (( fail )); then st "a shard failed, see $L/*.log"; echo "shard failure" > "$L/ERROR"; exit 1; fi
st "shards done; report"
if .venv/bin/python experiments/op_common_cause/scripts/cc_report.py > "$L/report.log" 2>&1; then
  st "done"; date > "$L/DONE"
else
  st "report failed"; tail -20 "$L/report.log" > "$L/ERROR"; exit 1
fi
