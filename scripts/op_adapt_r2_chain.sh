#!/usr/bin/env bash
# op-adapt r2 whole staged chain, one self-advancing script (prereg v5; sections 5 and 6):
#   gate (rescore-b DONE and V2 / V3-P5 / V5b / V3-V4 all pass) -> M1 (O readouts: eval, rater, read, navsim navtest)
#   -> stage1 -> stage10 -> full (lambda_s selection, every arm x seed).
# Idempotent: every step leaves a marker in $R/chain/ and is skipped on a re-run, so after a box restart just start the
# same command again (`scripts/tmux_run.sh r2-chain scripts/op_adapt_r2_chain.sh`). A failed gate or checklist writes
# $R/chain/ERROR (a stop line: report to main, do not continue); the end writes $R/chain/DONE. Logs: $R/chain/log.txt,
# events.jsonl, STATUS. GPUS and CORES are re-read from $R/chain/GPUS and $R/chain/CORES at every stage (edit them to move the
# chain; defaults 1,2 and 48-95). Stop processes by exact PID only.
set -uo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_CORETYPE=Haswell
R=$DATA_DIR/runs/op_adapt_r2
C=$R/chain
mkdir -p "$C"
PY=$DATA_DIR/envs/jevdrive/bin/python
TR=$DATA_DIR/envs/op-train/bin/python
[[ -x $TR ]] || TR=$PY
[[ -f $C/GPUS ]] || echo "1,2" > "$C/GPUS"
[[ -f $C/CORES ]] || echo "48-95" > "$C/CORES"
rm -f "$C/ERROR" "$C/DONE"
log() { echo "$(date +%F' '%T) $*" | tee -a "$C/log.txt"; }
ev() { printf '{"t": %s, "kind": "%s", "step": "%s"%s}\n' "$(date +%s)" "$1" "$2" "${3:-}" >> "$C/events.jsonl"; }
die() { log "ERROR: $*"; echo "$*" > "$C/ERROR"; ev error "$1"; exit 1; }
gpus() { cat "$C/GPUS"; }
cores() { cat "$C/CORES"; }
first_gpu() { gpus | cut -d, -f1; }
step() {          # step <name> <command...>: run once, marker on success
  local n=$1; shift
  [[ -f $C/$n.ok ]] && { log "skip $n (done)"; return 0; }
  echo "$n" > "$C/STATUS"; ev start "$n"; log "start $n"
  "$@" >> "$C/$n.log" 2>&1 || die "$n failed (see $C/$n.log)"
  date +%F' '%T > "$C/$n.ok"; ev end "$n"; log "done $n"
}
lane_stage() {    # lane_stage <stage> [extra lane args]: DONE = checklist passed, ERROR = stop line
  local s=$1; shift
  [[ -f $R/stage/$s/DONE ]] && return 0
  "$TR" scripts/op_adapt_r2_lane.py "$s" --gpus "$(gpus)" --cores "$(cores)" "$@"
  [[ -f $R/stage/$s/DONE ]] || { cat "$R/stage/$s/ERROR" 2>/dev/null; return 1; }
}

echo gate > "$C/STATUS"; log "chain start"
# ---- gate: the v5 re-score chain must be finished, every registered check that still applies must pass
B=$R/logs/rescore-b
until [[ -f $B/DONE || -f $B/ERROR ]]; do sleep 60; done
[[ -f $B/ERROR ]] && die "rescore-b failed"
"$PY" - <<'PYEOF' || die "gate: a registered check does not pass (see gate.json)"
import json, os, sys
R = os.path.expandvars("$DATA_DIR/runs/op_adapt_r2/checks/")
ld = lambda f: json.load(open(R + f))
v346, v3p5, v2, v5b, v5a = ld("V346.json"), ld("V3_p5.json"), ld("V2.json"), ld("V5b.json"), ld("V5a.json")
g = {"V3_sim": all(v346[d]["V3"]["pass"] for d in ("simC", "simK")), "V4": all(v346[d]["V4"]["pass"] for d in ("simC", "simK")),
     "V3_p5": v3p5["V3_p5"]["pass"], "V2": v2["pass"], "V5b": v5b["pass"], "V5a": bool(v5a.get("pass", v5a.get("all_pass", True)))}
json.dump(g, open(R + "gate.json", "w"), indent=1)
print(g)
sys.exit(0 if all(g.values()) else 1)
PYEOF
log "gate passed"

# ---- M1: zero-training baseline of the original model on every readout (before any training)
step m1_eval  env CUDA_VISIBLE_DEVICES="$(first_gpu)" taskset -c "$(cores)" "$TR" scripts/op_adapt_r2_readout.py eval --model O
step m1_rater env CUDA_VISIBLE_DEVICES="$(first_gpu)" taskset -c "$(cores)" "$TR" scripts/op_adapt_r2_readout.py rater --workers 8
step m1_read  env CUDA_VISIBLE_DEVICES="$(first_gpu)" taskset -c "$(cores)" "$TR" scripts/op_adapt_r2_readout.py read --model O
step m1_nav   env CUDA_VISIBLE_DEVICES="$(first_gpu)" "$TR" scripts/op_adapt_r2_readout.py navsim --split navtest --models O --cpus "$(cores)"

# ---- staged launch: 1 unit -> ~10 units -> full; a failed checklist is a stop line
step stage1  lane_stage stage1 --full-forward
step stage10 lane_stage stage10
step full    lane_stage full
echo end > "$C/STATUS"; date +%F' '%T > "$C/DONE"; ev end chain; log "chain done"
