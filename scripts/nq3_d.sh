#!/usr/bin/env bash
# Night queue 3, lane D (CPU first): Q4a -> Q4b -> Q5 -> Q6 as one unattended chain
# (todos/2026-09-26-night-queue-3.md, "一次性脚本与轮询" and the [D] entries).
#
# Resources (fixed by main): cores 180-199 (taskset), BLAS / OMP threads <= 20, GPU 6 with <= 20 GB.
# Every step writes runs/nq3/d/<step>/{log.txt,DONE}; the chain skips steps with DONE (restart = resume), pulls the repo
# before each step, stops a step at 2x its estimate and writes runs/nq3/d/ERROR (reason, last 50 log lines). Cross-lane
# hand-off: runs/nq3/q4a/PASS (lane B adds the mc_real0 arm). STATUS.md every 10 min, events.jsonl per step.
#   scripts/tmux_run.sh nq3-d scripts/nq3_d.sh
set -uo pipefail
: "${DATA_DIR:?}"
repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo"
D=$DATA_DIR/runs/nq3/d
mkdir -p "$D"
rm -f "$D/ERROR"
export CPUS=${D_CPUS:-180-199} GPU=${D_GPU:-6}
export OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20 NUMBA_NUM_THREADS=20 P5_SET=carla_p5v1_ba
PY=(taskset -c "$CPUS" "$repo/.venv/bin/python")

ev() { printf '{"t": %s, "kind": "%s", "step": "%s"%s}\n' "$(date +%s)" "$1" "$2" "${3:+, $3}" >> "$D/events.jsonl"; }

status() {   # STATUS.md: current step, elapsed / estimate, GPU 6, lane cores, container threads
  while sleep 600; do
    local cur est t0 now
    cur=$(cat "$D/current" 2>/dev/null || echo idle); est=$(cat "$D/$cur/est_min" 2>/dev/null || echo "?")
    t0=$(cat "$D/$cur/t0" 2>/dev/null || date +%s); now=$(date +%s)
    {
      echo "# lane D status ($(date '+%F %T %Z'))"
      echo "- step: $cur, elapsed $(( (now - t0) / 60 )) min of est. $est min (stop at 2x)"
      echo "- done: $(cd "$D" && ls -d */DONE 2>/dev/null | cut -d/ -f1 | tr '\n' ' ')"
      echo "- GPU $GPU: $(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
      echo "- cores $CPUS busy: $(ps -eo psr=,pcpu= | awk -v a="${CPUS%-*}" -v b="${CPUS#*-}" '$1>=a && $1<=b {s+=$2} END {printf "%.0f%%", s}')"
      echo "- container threads: $(cat /sys/fs/cgroup/pids.current) / $(cat /sys/fs/cgroup/pids.max)"
      echo "- last log line: $(tail -n 1 "$D/$cur/log.txt" 2>/dev/null | cut -c1-200)"
    } > "$D/STATUS.md.tmp" && mv "$D/STATUS.md.tmp" "$D/STATUS.md"
  done
}

fail() {     # step reason
  { echo "step: $1"; echo "reason: $2"; echo "time: $(date '+%F %T %Z')"; echo "--- last 50 log lines"; tail -n 50 "$D/$1/log.txt" 2>/dev/null; } > "$D/ERROR"
  ev error "$1" "\"reason\": \"$2\""
  echo "ERROR in $1: $2"
  kill "$status_pid" 2>/dev/null
  exit 1
}

step() {     # name est_min command...  (the command runs in its own process group, killed as a group on timeout)
  local name=$1 est=$2; shift 2
  [[ -f $D/$name/DONE ]] && { echo "[$(date +%T)] $name already done"; return 0; }
  git pull --ff-only -q || echo "git pull failed (continuing with the checked-out code)"
  mkdir -p "$D/$name"; echo "$name" > "$D/current"; echo "$est" > "$D/$name/est_min"; date +%s > "$D/$name/t0"
  ev step_start "$name" "\"est_min\": $est"
  echo "[$(date +%T)] $name (est. $est min): $*"
  local t0; t0=$(date +%s)
  setsid "$@" > "$D/$name/log.txt" 2>&1 &
  local pid=$!
  echo "$pid" > "$D/$name/pid"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 20
    if (( $(date +%s) - t0 > 2 * est * 60 )); then
      kill -TERM -- "-$pid" 2>/dev/null; sleep 10; kill -KILL -- "-$pid" 2>/dev/null
      fail "$name" "exceeded 2x the estimate ($est min)"
    fi
  done
  wait "$pid"; local rc=$?
  (( rc == 0 )) || fail "$name" "exit code $rc"
  local wall=$(( $(date +%s) - t0 ))
  printf 'step: %s\nwall_s: %s\nfinished: %s\nlog: %s\n' "$name" "$wall" "$(date '+%F %T %Z')" "$D/$name/log.txt" > "$D/$name/DONE"
  ev step_end "$name" "\"wall_s\": $wall"
  echo "[$(date +%T)] $name done in $(( wall / 60 )) min"
}

need() {     # wait (pulling every 5 min, <= 3 h) for a step script another session is still writing
  local f=$1 t0; t0=$(date +%s)
  until [[ -x $f ]]; do
    (( $(date +%s) - t0 > 10800 )) && fail "$(basename "$f" .sh)" "$f not written within 3 h"
    echo "[$(date +%T)] waiting for $f"; sleep 300; git pull --ff-only -q || true
  done
}

status & status_pid=$!
ev start chain

q=("${PY[@]}" -m jevdrive.nq3_q4a)
# ---- Q4a
step q4a-prep 15 "${q[@]}" prep
step q4a-qwen 120 bash -c "for i in 0 1; do CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 taskset -c $CPUS $repo/.venv/bin/python -m jevdrive.navsim_qwen work nq3_navtrain2 --workers 3 & sleep 60; done; wait; $repo/.venv/bin/python -m jevdrive.navsim_qwen check nq3_navtrain2"
step q4a-fit 45 env CUDA_VISIBLE_DEVICES="$GPU" "${q[@]}" fit
step q4a-hydra 20 env CUDA_VISIBLE_DEVICES="$GPU" "${q[@]}" hydra
step q4a-hold 40 bash -c "${q[*]} hold-jobs && taskset -c $CPUS scripts/nq3_d/navscore.sh $DATA_DIR/runs/nq3/q4a/hold/jobs.txt 2 9 && ${q[*]} select"
step q4a-nav 45 bash -c "${q[*]} nav-jobs && taskset -c $CPUS scripts/nq3_d/navscore.sh $DATA_DIR/runs/nq3/q4a/navtest/jobs.txt 2 9 && CUDA_VISIBLE_DEVICES=$GPU ${q[*]} report"
# ---- Q4b, Q5, Q6 (entry scripts take no arguments; each prints its own estimate)
for s in q4b:90 q5:240 q6:210; do
  n=${s%%:*} est=${s#*:}
  need "$repo/scripts/nq3_d/$n.sh"
  step "$n" "$est" "$repo/scripts/nq3_d/$n.sh"
done
date '+%F %T %Z' > "$D/DONE"
ev end chain
kill "$status_pid" 2>/dev/null
echo "lane D done"
