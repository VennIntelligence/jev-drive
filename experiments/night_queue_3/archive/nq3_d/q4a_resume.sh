#!/usr/bin/env bash
# Resume path for night queue 3 Q4a after the user-authorized deviation of 2026-09-27 20:50 (tolerance of the lam_r = 0
# check 1e-3 -> 1e-2 m plus exact P5 exam cells; fc65452:tmp/2026-09-27-q4a-fp64.md). Same steps, commands
# and state layout as experiments/night_queue_3/archive/nq3_d.sh (runs/nq3/d/<step>/{log.txt,DONE}, stop at 2x the estimate -> runs/nq3/d/ERROR),
# but only the Q4a steps, no git pull inside, and staged: run the stages given on the command line, in order.
#   stages: fit hydra hold-pilot hold nav-pilot nav
#   hold-pilot / nav-pilot score only the first job of the list (the sanity unit); hold / nav score the rest (scored
#   jobs are skipped by navscore.sh), then `select` (lam_select.json, before any navtest score) / `report`.
#   Q4A_CPUS (core list), Q4A_GPU, Q4A_THREADS (BLAS), Q4A_PAR x Q4A_NT (devkit jobs x ray threads) override the defaults.
#   scripts/tmux_run.sh q4a-resume experiments/night_queue_3/archive/nq3_d/q4a_resume.sh fit hydra
set -uo pipefail
: "${DATA_DIR:?}"
repo=$(cd "$(dirname "$0")/../../../.." && pwd)
cd "$repo"
D=$DATA_DIR/runs/nq3/d
Q=$DATA_DIR/runs/nq3/q4a
mkdir -p "$D"
[[ -f $D/ERROR ]] && { echo "$D/ERROR exists: archive it first"; exit 1; }
CPUS=${Q4A_CPUS:-64-103} GPU=${Q4A_GPU:-6} TH=${Q4A_THREADS:-20} PAR=${Q4A_PAR:-4} NT=${Q4A_NT:-10}
export OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH NUMBA_NUM_THREADS=$TH P5_SET=carla_p5v1_ba
q="taskset -c $CPUS $repo/.venv/bin/python -m experiments.night_queue_3.archive.nq3_q4a"
score="taskset -c $CPUS experiments/night_queue_3/archive/nq3_d/navscore.sh"

ev() { printf '{"t": %s, "kind": "%s", "step": "%s"%s}\n' "$(date +%s)" "$1" "$2" "${3:+, $3}" >> "$D/events.jsonl"; }

fail() {
  { echo "step: $1"; echo "reason: $2"; echo "time: $(date '+%F %T %Z')"; echo "--- last 50 log lines"; tail -n 50 "$D/$1/log.txt" 2>/dev/null; } > "$D/ERROR"
  ev error "$1" "\"reason\": \"$2\""
  echo "ERROR in $1: $2"
  exit 1
}

step() {     # name est_min command (one string, run by bash in its own process group)
  local name=$1 est=$2 cmd=$3
  [[ -f $D/$name/DONE ]] && { echo "[$(date +%T)] $name already done"; return 0; }
  mkdir -p "$D/$name"; echo "$name" > "$D/current"; echo "$est" > "$D/$name/est_min"; date +%s > "$D/$name/t0"
  ev step_start "$name" "\"est_min\": $est, \"path\": \"q4a_resume\""
  echo "[$(date +%T)] $name (est. $est min): $cmd"
  local t0; t0=$(date +%s)
  setsid bash -c "$cmd" > "$D/$name/log.txt" 2>&1 &
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
  printf 'step: %s\nwall_s: %s\nfinished: %s\nlog: %s\npath: q4a_resume\n' "$name" "$wall" "$(date '+%F %T %Z')" "$D/$name/log.txt" > "$D/$name/DONE"
  ev step_end "$name" "\"wall_s\": $wall"
  echo "[$(date +%T)] $name done in $(( wall / 60 )) min"
}

for s in "$@"; do
  case $s in
    fit)        step q4a-fit 45 "CUDA_VISIBLE_DEVICES=$GPU $q fit" ;;
    hydra)      step q4a-hydra 20 "CUDA_VISIBLE_DEVICES=$GPU $q hydra" ;;
    hold-pilot) step q4a-hold-pilot 20 "$q hold-jobs && head -n 1 $Q/hold/jobs.txt > $Q/hold/jobs_pilot.txt && NAVSIM_THREADS=$NT $score $Q/hold/jobs_pilot.txt 1 $NT" ;;
    hold)       step q4a-hold 40 "$score $Q/hold/jobs.txt $PAR $NT && $q select" ;;
    nav-pilot)  step q4a-nav-pilot 40 "test -f $Q/lam_select.json && $q nav-jobs && head -n 1 $Q/navtest/jobs.txt > $Q/navtest/jobs_pilot.txt && $score $Q/navtest/jobs_pilot.txt 1 $NT" ;;
    nav)        step q4a-nav 90 "$score $Q/navtest/jobs.txt $PAR $NT && CUDA_VISIBLE_DEVICES=$GPU $q report" ;;
    *) echo "unknown stage $s"; exit 1 ;;
  esac
done
echo "stages done: $*"
