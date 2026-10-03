#!/usr/bin/env bash
# Launch-lean lane (plans/2026-10-04-launch-lean-prereg.md): the four lean_probe.py jobs on one card, in parallel, resumable
# (a job whose output exists is skipped). Jobs files come from the lane's setup ($L/jobs64.json, jobs20.json, jobs_replay.json).
# Usage (box, tmux):  GPU=2 CPUS=150-199 experiments/hugsim/scripts/lean_chain.sh
# Files: $DATA_DIR/runs/hugsim-lean/{STATUS,DONE,ERROR,<job>.json,<job>.log}
set -uo pipefail
: "${DATA_DIR:?}" "${GPU:?}" "${CPUS:?}"
cd "$(dirname "$0")/../../.."
L=$DATA_DIR/runs/hugsim-lean
PY=$DATA_DIR/envs/openpilot/bin/python
P=experiments/hugsim/scripts/lean_probe.py
rm -f "$L/DONE" "$L/ERROR"
st() { echo "$(date '+%F %T') $*" | tee "$L/STATUS"; }
job() {   # name, args...
  local n=$1; shift
  [[ -f $L/$n.json ]] && return 0
  CUDA_VISIBLE_DEVICES=$GPU taskset -c "$CPUS" $PY $P "$@" --out "$L/$n.json" > "$L/$n.log" 2>&1
}
st "running lean_O64, lean_adapt20, rate20, replay"
job lean_O64 lean "$L/jobs64.json" --models O & p1=$!
job lean_adapt20 lean "$L/jobs20.json" --models pilot-s0 it_dw3-s0 --variants base,mirror,mirror_tc,tc,single,roll & p2=$!
job rate20 rate "$L/jobs20.json" --models O pilot-s0 it_dw3-s0 --k 6 & p3=$!
job replay replay "$L/jobs_replay.json" --models O pilot-s0 it_dw3-s0 --steps 20 & p4=$!
bad=0
for p in $p1 $p2 $p3 $p4; do wait $p || bad=1; done
(( bad )) && { st "a job failed"; cp "$L/STATUS" "$L/ERROR"; exit 1; }
st "done"
touch "$L/DONE"
