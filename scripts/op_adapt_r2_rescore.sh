#!/usr/bin/env bash
# op-adapt r2, prereg v5: re-score with the v5 P rule (moving blocker -> P = 1, static dwell gate T_w = 5 s) and re-run the
# score-table checks. Old tables and checks are moved (never deleted) to score/run1 and checks/run1_v346 first (once).
# Usage (box): scripts/tmux_run.sh r2-rescore-a scripts/op_adapt_r2_rescore.sh a     # simC, simK, V3/V4
#              scripts/tmux_run.sh r2-rescore-b scripts/op_adapt_r2_rescore.sh b     # nus, p5, V3-p5, V2, sanity
# Env: CPUS (taskset list, default 96-119), WORKERS (default 24). Writes $DATA_DIR/runs/op_adapt_r2/logs/rescore-<phase>/.
set -uo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_CORETYPE=Haswell
phase=${1:?phase a|b}
R=$DATA_DIR/runs/op_adapt_r2
D=$R/logs/rescore-$phase
mkdir -p "$D"
rm -f "$D/DONE" "$D/ERROR"
PY=$DATA_DIR/envs/jevdrive/bin/python
CPUS=${CPUS:-96-119}
W=${WORKERS:-24}
ev() { printf '{"t": %s, "kind": "%s", "step": "%s"%s}\n' "$(date +%s)" "$1" "$2" "${3:-}" >> "$D/events.jsonl"; }
if [[ $phase == a && ! -d $R/score/run1 ]]; then
  mkdir -p "$R/score/run1" "$R/checks/run1_v346"
  for d in simC simK nus off p5; do mv "$R/score/$d.npz" "$R/score/$d.json" "$R/score/run1/" 2>/dev/null; done
  mv "$R/score/sanity.json" "$R/score/run1/" 2>/dev/null
  mv "$R/checks/V346.json" "$R/checks/V3_p5.json" "$R/checks/V2.json" "$R/checks/V2_frames.parquet" "$R/checks/V5b.json" "$R/checks/run1_v346/" 2>/dev/null
fi
if [[ $phase == a ]]; then steps=("score simC" "score simK" "v346"); else steps=("score nus" "score p5" "v3-p5" "v2" "sanity"); fi
ev start "$phase"
for s in "${steps[@]}"; do
  set -- $s
  echo "$(date +%F' '%T) step $s" | tee -a "$D/log.txt"; echo "$s" > "$D/STATUS"; ev step_start "$1"
  extra=(); [[ $1 == score ]] && extra=(--domain "$2")
  if ! taskset -c "$CPUS" "$PY" -m jevdrive.op_adapt_score_data "$1" "${extra[@]}" --workers "$W" 2>&1 | tee -a "$D/log.txt"; then
    ev end "$s" ', "ok": false'; echo "$s" > "$D/ERROR"; exit 1
  fi
  ev step_end "$1"
done
ev end "$phase" ', "ok": true'
date +%F' '%T > "$D/DONE"
