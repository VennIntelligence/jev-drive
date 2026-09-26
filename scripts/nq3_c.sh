#!/usr/bin/env bash
# Night queue 3, lane C chain (todos/2026-09-26-night-queue-3.md, "一次性脚本与轮询", Q1 / Q2, the [C] entries).
# One-shot: every step checks the previous step's DONE, runs under a timeout of twice its estimate, writes
# runs/nq3/c/<step>/DONE (wall, outputs) or runs/nq3/c/ERROR (reason, last 50 log lines, progress) and stops the lane.
# Resources: GPU 6 only, cores 150-179 (sub-steps pin their own share), threads <= 30.
# Usage (box, tmux jev:nq3-c): scripts/nq3_c.sh            # the whole lane, resumes after the last DONE
#                              scripts/nq3_c.sh --only a,b  # just these steps (used before the chain started)
set -uo pipefail
: "${DATA_DIR:?}"
cd "$(dirname "$0")/.."
REPO=$PWD
R=$DATA_DIR/runs/nq3/c
mkdir -p "$R"
echo "$$ chain $(date '+%F %T')" >> "$R/pids.txt"
ONLY=""
[[ ${1:-} == --only ]] && ONLY=",$2,"
EV=$R/events.jsonl
CUR=$R/current_step

ev() { printf '{"t": %s, "kind": "%s"%s}\n' "$(date +%s.%N | cut -c1-14)" "$1" "${2:+, $2}" >> "$EV"; }

status_loop() {
  while true; do
    {
      echo "# lane C status ($(date '+%F %T %Z'))"
      echo
      echo "current step: $(cat "$CUR" 2>/dev/null || echo none)"
      echo
      echo "| step | state | wall (min) | estimate (min) |"
      echo "|:--|:--|--:|--:|"
      for s in "${STEPS[@]}"; do
        n=${s%%:*}; est=${s#*:}
        if [[ -f $R/$n/DONE ]]; then st=done; w=$(sed -n 's/.*"wall_min": \([0-9.]*\).*/\1/p' "$R/$n/DONE")
        elif [[ -f $R/$n/started ]]; then st=running; w=$(( ($(date +%s) - $(cat "$R/$n/started")) / 60 ))
        else st=pending; w=""; fi
        echo "| $n | $st | $w | $est |"
      done
      echo
      echo "progress of the running step (last log line): $(tail -c 400 "$R/$(cat "$CUR" 2>/dev/null)/log.txt" 2>/dev/null | tr '\r' '\n' | grep -v '^$' | tail -1)"
      echo
      echo "GPU 6: $(nvidia-smi -i 6 --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
      echo "load: $(cut -d' ' -f1-3 /proc/loadavg); pids.current $(cat /sys/fs/cgroup/pids.current 2>/dev/null)"
      echo "lane C processes on cores 150-179: $(ps -eo psr,pid | awk '$1>=150 && $1<=179' | wc -l)"
    } > "$R/STATUS.md.tmp" && mv "$R/STATUS.md.tmp" "$R/STATUS.md"
    sleep 600
  done
}

fail() {   # step, reason
  local s=$1
  { echo "step: $s"; echo "reason: $2"; echo "time: $(date '+%F %T %Z')";
    echo "done steps: $(ls -d "$R"/*/DONE 2>/dev/null | xargs -n1 dirname 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')";
    echo "--- last 50 log lines ---"; tail -n 50 "$R/$s/log.txt" 2>/dev/null | tr '\r' '\n' | tail -n 50; } > "$R/ERROR"
  ev error "\"step\": \"$s\", \"reason\": \"$2\""
  echo "ERROR in $s: $2"
  exit 1
}

# step <name> <estimate_min> <command...>: skip if DONE; else wait for the previous steps' DONE, run under a timeout
# of 2 x estimate, log to <step>/log.txt, DONE with the wall time and the step's outputs.json (if it wrote one).
step() {
  local n=$1 est=$2; shift 2
  [[ -n $ONLY && $ONLY != *",$n,"* ]] && return 0
  [[ -f $R/$n/DONE ]] && { echo "skip $n (DONE)"; return 0; }
  mkdir -p "$R/$n"
  exec {LK}>"$R/$n/lock"; flock "$LK"           # another chain instance running this step: wait for it
  [[ -f $R/$n/DONE ]] && { echo "skip $n (DONE by another instance)"; exec {LK}>&-; return 0; }
  date +%s > "$R/$n/started"; echo "$n" > "$CUR"
  ev step_start "\"step\": \"$n\", \"estimate_min\": $est"
  echo "$(date '+%T') start $n (estimate $est min)"
  local t0=$SECONDS
  timeout --signal=INT --kill-after=120 $((2 * est * 60)) bash -c "source '$REPO/scripts/nq3_c_steps.sh'; $*" \
    > >(tee -a "$R/$n/log.txt") 2>&1
  local rc=$?
  local w; w=$(awk "BEGIN{printf \"%.1f\", ($SECONDS - $t0) / 60}")
  if (( rc == 124 )); then fail "$n" "wall ${w} min exceeded twice the estimate ($est min)"; fi
  (( rc == 0 )) || fail "$n" "exit code $rc after ${w} min"
  printf '{"step": "%s", "wall_min": %s, "estimate_min": %s, "finished": "%s", "outputs": %s}\n' "$n" "$w" "$est" \
    "$(date '+%F %T %Z')" "$(cat "$R/$n/outputs.json" 2>/dev/null || echo null)" > "$R/$n/DONE"
  ev step_end "\"step\": \"$n\", \"wall_min\": $w"
  exec {LK}>&-
  echo "$(date '+%T') done $n in $w min"
}

STEPS=("feats:90" "q1_op:25")

status_loop & SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT
ev start "\"only\": \"$ONLY\""
step feats 90 feats
step q1_op 25 q1_op
[[ -z $ONLY ]] && { date '+%F %T %Z' > "$R/DONE"; ev end; }
echo "lane C: finished ($ONLY)"
