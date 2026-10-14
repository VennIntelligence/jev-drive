#!/usr/bin/env bash
# P4 box determinism and packing probe (results/p4_self/box_det.md). One simulator launch of N identity sessions of one scene (P4's identity
# cell: the branch switch off), one rollout at a time, through the launch path of rf_p4.sh's `stack` as of P4 Amendment 1 (P4_CONC=1):
# box = pai_native.sh (the card the GPU pool gives), tokyo = pai_run.sh (Docker, card $CARD). Serving = the reference
# /data/runs/route_night/r281/rsg (RSG-L18-s0, JEV_VCONT=1.0 JEV_LEAD=1, gains 20) plus P4's send_recording_ground_truth=true and PAI_DUMP.
#   rf_p4det.sh <root> <arm = id8-tag> <N sessions>   -> <root>/<arm>/{sim/, driver -> sim/driver-logs (box), timing.json}
# Env: REPO (served tree; Tokyo: the frozen P4 clone /data/runs/route_night/p4/code), CARD (Tokyo card, 1).
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); repo=${REPO:-$(cd "$here/../../.." && pwd)}
arm=$2 n=$3; r=$(mkdir -p "$1" && cd "$1" && pwd)/$arm; D=${DATA_DIR:-/data}; T=RSG-L18-s0; CARD=${CARD:-1}
BACKEND=${BACKEND:-$([[ -d /data/runs/route_night ]] && echo tokyo || echo box)}
if [[ $BACKEND == tokyo ]]; then PY=python3; else PY=$repo/.venv/bin/python; fi
WIZ=(runtime.simulation_config.send_recording_ground_truth=true controller.gains.rel_front_steering_angle_weight=20 "runtime.simulation_config.n_rollouts=$n")
rm -rf "$r" "$r.out"; mkdir -p "$r"
$PY - "$repo/experiments/route_follow/results/p4_self/plan.json" "${arm:0:8}" "$n" "$r" <<'PY' || exit 1
import json, sys
P = json.load(open(sys.argv[1]))["scenes"]; sc = next(s for s in P if s[7:15] == sys.argv[2]); n = int(sys.argv[3])
k = ("u", "x", "y", "kappa", "s_entry", "s_exit", "s_release", "v_e")                 # rf_p4.sh smoke's S1 cell file
c = dict(tag="id", D=None, src="self", g=1.0, lon="raw", switch=0)
json.dump(dict(scenes={sc: {x: P[sc][x] for x in k}}, cells={sc: [c] * n}), open(f"{sys.argv[4]}/p4cells.json", "w"))
open(f"{sys.argv[4]}/scene.txt", "w").write(sc + "\n")
PY
t0=$(date +%s.%N)
if [[ $BACKEND == tokyo ]]; then
  mkdir -p "$r/driver"; chmod 777 "$r/driver"; cp "$r/p4cells.json" "$r/driver/p4cells.json"; L=$repo/experiments/alpasim/lib
  for f in onp_log_driver.py onp_p4.py; do cmp -s "$repo/experiments/onpol_pai/lib/$f" "$L/$f" || { echo "$L/$f differs from onpol_pai/lib" >&2; exit 1; }; done
  GPU=$CARD CONC=1 PAI_BASEPORT=$((6400 + 100 * CARD)) TAG=$T DRV_PY=onp_log_driver.py \
    DRV_ENV="-e JEV_VCONT=1.0 -e JEV_LEAD=1 -e PAI_DUMP=100000 -v $L/onp_p4.py:/app/jev-drive/experiments/alpasim/lib/onp_p4.py:ro -v $D/runs/op_parity/runs/$T:/app/data/runs/op_parity/runs/$T:ro" \
    bash "$repo/experiments/alpasim/scripts/pai_run.sh" "$r" "$r/scene.txt" "${WIZ[@]}" > "$r.out" 2>&1; rc=$?
else
  mkdir -p "$r/sim/driver-logs"; cp "$r/p4cells.json" "$r/sim/driver-logs/p4cells.json"; ln -sfn sim/driver-logs "$r/driver"
  env SH30_TAG="$T" JEV_VCONT=1.0 JEV_LEAD=1 PAI_DUMP=100000 DRV_PY=../../onpol_pai/lib/onp_log_driver.py CONC=1 \
    bash "$repo/experiments/alpasim/scripts/pai_native.sh" "$r/sim" "$r/scene.txt" "${WIZ[@]}" > "$r.out" 2>&1; rc=$?
  rm -rf "$r/sim/nre-cache" "$r/sim/driver-tmp"
fi
t1=$(date +%s.%N)
find "$r/sim/rollouts" -name '*.mp4' -delete 2>/dev/null
$PY - "$r" "$n" "$rc" "$t0" "$t1" "$BACKEND" <<'PY'
import glob, json, os, sys
r, n, rc, t0, t1, be = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
f = f"{r}/sim/aggregate/results-summary.json"
rows = json.load(open(f))["rollouts"] if os.path.exists(f) else []
ok = [x for x in rows if "error" not in (x.get("metrics") or {})]
st = sorted(json.loads(l)["t"] for l in open(f"{r}/driver/start.jsonl")) if os.path.exists(f"{r}/driver/start.jsonl") else []
asl = [os.path.getsize(p) for p in glob.glob(f"{r}/sim/rollouts/*/*/rollout.asl")]
out = dict(arm=os.path.basename(r), backend=be, n=n, rc=rc, scored=len(ok), t0=t0, t1=t1, wall_s=round(t1 - t0, 1), session_starts=st,
           gpu=os.environ.get("CL_GPU"), job=os.environ.get("CL_POOL_JOB"), asl_bytes=asl)
json.dump(out, open(f"{r}/timing.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("arm", "rc", "scored", "wall_s", "gpu")}))
sys.exit(0 if rc == 0 and len(ok) == n else 1)
PY
