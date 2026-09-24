#!/usr/bin/env bash
# Alpamayo 1.5 zero-shot Bench2Drive exam with the Bench2DriveZoo PID (todos/2026-09-24-zeroshot-exam/bench2drive.md).
# One job = one resident Alpamayo policy server + b2d_run.py workers, then the server is stopped.
# Meant to run under scripts/slot_run.sh (which sets CUDA_VISIBLE_DEVICES from --gpu); CARLA takes the same card via
# --gpu-rank (Vulkan index == CUDA index on this box, checked by UUID 2026-09-25).
#
#   scripts/zeroshot_b2d_alp.sh smoke <gpu>   5 pre-registered routes, 1 worker. Exits non-zero on INFRASTRUCTURE
#                                             failure only: runner error, server crash/restart, route never finished,
#                                             harness error in a route, policy server gone. Driving outcome never.
#   scripts/zeroshot_b2d_alp.sh full <gpu>    220 routes, 4 workers, resumable (rerun skips done/<id>.json).
#                                             Non-zero if the policy server dies, there is no summary, or more than
#                                             15 routes never finished (11 is the known CARLA-crash baseline).
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
mode=$1 gpu=$2
D=$DATA_DIR/runs/zeroshot-exam/b2d
case $mode in
    smoke) out=$D/smoke-alpamayo-zoopid; workers=1; sidx=400; routes=(--route-ids 2390,24211,1711,2373,3564) ;;
    full)  out=$D/full220-alpamayo-zoopid; workers=4; sidx=410; routes=(--towns all) ;;
    *) echo "mode must be smoke or full" >&2; exit 2 ;;
esac
sock=$D/alpamayo-$mode-zoopid.sock ready=$D/alpamayo-$mode-zoopid.ready cfg=$D/agent-alpamayo-$mode-zoopid.json
rm -f "$sock" "$ready"
cat > "$cfg" <<EOF
{"model": "alpamayo", "socket": "$sock", "plan_every": 5, "controller": "zoo_pid", "controller_preset": "carla",
 "controller_config": "$(pwd)/todos/2026-09-22-b2d-controller/results/controller_config.json",
 "seed": 0, "dump_every": $([[ $mode == smoke ]] && echo 1 || echo 0)}
EOF

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 \
    "$DATA_DIR/third_party/alpamayo1.5/.venv/bin/python" scripts/zeroshot_policy_server.py alpamayo \
    --socket "$sock" --ready-file "$ready" > "$D/alpamayo-$mode-zoopid.server.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null; wait $server 2>/dev/null' EXIT
until [[ -e $ready ]]; do
    kill -0 $server 2>/dev/null || { echo "policy server died during start-up, see $D/alpamayo-$mode-zoopid.server.log"; exit 3; }
    sleep 5
done
echo "$(date +%T) policy server ready (pid $server) on GPU $gpu"

"$DATA_DIR/envs/carla/bin/python" scripts/b2d_run.py "${routes[@]}" --workers $workers --server-index $sidx \
    --gpu-rank "$gpu" --python "$DATA_DIR/envs/b2d-tcp/bin/python" --agent scripts/b2d_zeroshot_agent.py \
    --agent-config "$cfg" --decimate 2 --no-spectator --max-attempts 2 --out "$out"
rc=$?
kill -0 $server 2>/dev/null || { echo "policy server died during the run"; exit 3; }
if [[ $mode == full ]]; then
    # b2d_run exits 1 when any route never finished; the previous full round lost 11 Town12/13 routes to CARLA
    # segfaults (docs/carla.md), which is reported with the score, not an infrastructure failure of this job.
    python3 -c "import json,sys; s=json.load(open('$out/summary.json')); n=len(s['routes_never_finished']); \
print('never finished:', s['routes_never_finished']); sys.exit(0 if n <= 15 else 1)" || { echo "runner exit $rc"; exit 1; }
    exit 0
fi
(( rc == 0 )) || { echo "runner exit $rc"; exit $rc; }

python3 - "$out" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
s = json.loads((out / "summary.json").read_text())
bad = []
if s.get("routes_never_finished"): bad.append("never finished: %s" % s["routes_never_finished"])
if s.get("restarts"): bad.append("server restarts: %s" % s["restarts"])
for rr in out.glob("attempts/*/*/route_result.json"):
    st = json.loads(rr.read_text()).get("status")
    if st != "finished": bad.append("%s: %s" % (rr.parent, st))
for d in sorted(out.glob("attempts/*/*")):
    if not (d / "plans.jsonl").exists() or not (d / "plans.jsonl").stat().st_size: bad.append("%s: no plans" % d)
print("smoke harness check:", "OK" if not bad else "; ".join(bad))
sys.exit(1 if bad else 0)
EOF
