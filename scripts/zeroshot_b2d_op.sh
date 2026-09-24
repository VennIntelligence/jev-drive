#!/usr/bin/env bash
# openpilot zero-shot Bench2Drive exam with the fixed adapter (todos/2026-09-24-zeroshot-exam/openpilot-migration.md):
# rear-axle plan origin, every-tick synced cameras, 5 s warm-up. Controllers: "zoo_pid" = Bench2DriveZoo's official
# UniAD/VAD PID run as shipped (primary, user decision 2026-09-25), "native" = openpilot's desired curvature / accel
# (paired secondary), "fixed" = the pre-registered repo controller (for the before/after comparison with the smoke).
# Run under scripts/slot_run.sh (it sets CUDA_VISIBLE_DEVICES); CARLA takes the same card via --gpu-rank.
#
#   scripts/zeroshot_b2d_op.sh smoke <gpu>   5 pre-registered routes x 5 phases, 2 workers:
#        zoo-lebowski (pre-registered model), zoo-cinque, native-cinque, fixed-lebowski (old controller + fixes),
#        shadow-cinque (route oracle drives, the model only plans: open-loop error on CARLA frames).
#        Non-zero only on infrastructure failure (runner error, route never finished, harness status, no plans,
#        policy server gone); driving outcome never.
#   scripts/zeroshot_b2d_op.sh full <gpu>    220 routes, 4 workers, resumable. Model and whether the native path also
#        gets a 220 run are decided from the smoke by the rule pre-registered in the doc (choose_full below).
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
mode=$1 gpu=$2
D=$DATA_DIR/runs/zeroshot-exam/b2d-op
C=$(pwd)/todos/2026-09-22-b2d-controller/results/controller_config.json
PY_OP=$DATA_DIR/envs/openpilot/bin/python
mkdir -p "$D"
servers=()
trap 'for p in "${servers[@]}"; do kill $p 2>/dev/null; done; wait 2>/dev/null' EXIT

start_server() {  # model -> socket path on stdout
    local m=$1 sock=$D/$1-$mode.sock ready=$D/$1-$mode.ready
    rm -f "$sock" "$ready"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PY_OP scripts/zeroshot_policy_server.py "$m" --socket "$sock" \
        --ready-file "$ready" > "$D/server-$m-$mode.log" 2>&1 &
    servers+=($!)
    local pid=$!
    until [[ -e $ready ]]; do
        kill -0 $pid 2>/dev/null || { echo "policy server $m died at start-up, see $D/server-$m-$mode.log" >&2; exit 3; }
        sleep 5
    done
    echo "$(date +%T) server $m ready (pid $pid)" >&2
}

config() {  # name model controller drive dump
    local plan_every=1
    [[ $2 == lebowski ]] && plan_every=4
    cat > "$D/agent-$1.json" <<EOF
{"model": "$2", "socket": "$D/$2-$mode.sock", "plan_every": $plan_every, "controller": "$3", "drive": "$4",
 "controller_preset": "carla", "controller_config": "$C", "seed": 0, "dump_every": $5,
 "op_camera_tick": 0.05, "plan_origin": "rear", "warmup_s": 5.0}
EOF
}

run() {  # phase workers server-index route-args...
    local ph=$1 w=$2 sidx=$3; shift 3
    echo "$(date '+%F %T') phase $ph" >&2
    "$DATA_DIR/envs/carla/bin/python" scripts/b2d_run.py "$@" --workers "$w" --server-index "$sidx" --gpu-rank "$gpu" \
        --python "$DATA_DIR/envs/b2d-tcp/bin/python" --agent scripts/b2d_zeroshot_agent.py \
        --agent-config "$D/agent-$ph.json" --decimate 4 --no-spectator --max-attempts 2 --out "$D/$mode-$ph"
}

harness_check() {  # out dir -> non-zero on infrastructure failure only
    python3 - "$1" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
s = json.loads((out / "summary.json").read_text())
bad = []
if s.get("routes_never_finished"): bad.append("never finished: %s" % s["routes_never_finished"])
for rr in out.glob("attempts/*/*/route_result.json"):
    st = json.loads(rr.read_text()).get("status")
    if st != "finished": bad.append("%s: %s" % (rr.parent.name, st))
for d in sorted(out.glob("attempts/*/*")):
    p = d / "plans.jsonl"
    if not p.exists() or not p.stat().st_size: bad.append("%s: no plans" % d)
print(out.name, "harness check:", "OK" if not bad else "; ".join(bad))
sys.exit(1 if bad else 0)
EOF
}

if [[ $mode == smoke ]]; then
    R=(--route-ids 2390,24211,1711,2373,3564)
    config zoo-lebowski lebowski zoo_pid model 5
    config zoo-cinque cinque zoo_pid model 20
    config native-cinque cinque native model 20
    config fixed-lebowski lebowski fixed model 5
    config shadow-cinque cinque fixed oracle 20
    start_server lebowski
    start_server cinque
    fail=0
    for ph in zoo-lebowski zoo-cinque native-cinque fixed-lebowski shadow-cinque; do
        run $ph 2 500 "${R[@]}" || { echo "phase $ph runner exit $?" >&2; fail=1; }
        for p in "${servers[@]}"; do kill -0 $p 2>/dev/null || { echo "a policy server died" >&2; exit 3; }; done
        harness_check "$D/smoke-$ph" || fail=1
    done
    exit $fail
fi

[[ $mode == full ]] || { echo "mode must be smoke or full" >&2; exit 2; }
# Pre-registered choice (openpilot-migration.md, "C2 全量"), made from the smoke only, before any full result:
#   model  = cinque, unless zoo-lebowski's smoke mean DS exceeds zoo-cinque's by >= 10 -> lebowski
#   native = also run the 220 routes with controller native only if native-cinque's smoke mean DS exceeds
#            zoo-cinque's by >= 10 (it then runs after the primary, same model)
read -r model native < <(python3 - "$D" <<'EOF'
import json, sys
from pathlib import Path
D = Path(sys.argv[1])
def ds(ph):
    v = [json.loads(p.read_text())["_checkpoint"]["records"][0]["scores"]["score_composed"]
         for p in (D / ("smoke-" + ph)).glob("attempts/*/*/results.json")]
    return sum(v) / len(v) if v else float("nan")
zl, zc, nc = ds("zoo-lebowski"), ds("zoo-cinque"), ds("native-cinque")
model = "lebowski" if zl >= zc + 10 else "cinque"
print(model, "yes" if nc >= zc + 10 else "no")
print("smoke mean DS: zoo-lebowski %.1f zoo-cinque %.1f native-cinque %.1f -> model %s" % (zl, zc, nc, model),
      file=sys.stderr)
EOF
)
echo "$(date '+%F %T') full: model=$model native_220=$native" | tee "$D/full-choice.txt" >&2
config full-zoo "$model" zoo_pid model 0
start_server "$model"
check_full() {
    python3 -c "import json,sys; s=json.load(open('$1/summary.json')); n=len(s['routes_never_finished']); \
print('never finished:', s['routes_never_finished']); sys.exit(0 if n <= 15 else 1)"
}
run full-zoo 4 520 --towns all
kill -0 "${servers[0]}" 2>/dev/null || { echo "policy server died during the run" >&2; exit 3; }
check_full "$D/full-full-zoo" || exit 1
if [[ $native == yes ]]; then
    config full-native "$model" native model 0
    run full-native 4 520 --towns all
    check_full "$D/full-full-native" || exit 1
fi
exit 0
