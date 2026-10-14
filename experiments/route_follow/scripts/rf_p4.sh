#!/usr/bin/env bash
# Turning school P4, self-branch search on the AlpaSim PAI simulator (prereg experiments/route_follow/plans/2026-10-14-p4-self-branch-prereg.md;
# rf_p4.py; driver experiments/onpol_pai/lib/onp_log_driver.py ONP_CELLS + onp_p4.py). One scene = one job: phase A = the policy alone
# (RSG-L18-s0, the reference serving) for one rollout; reach (handover decision per D from its own stations, or unreached -> log fallback);
# phase B = every branch cell of the scene in ONE simulator launch (n_rollouts = cells, 4 concurrent; the driver gives the n-th session of
# the scene cell n); then the scene's read. Backends: tokyo (Docker stack pai_run.sh on card $CARD, a frozen clone of this repo) and box
# (pai_native.sh, the card the GPU pool gives).
#   rf_p4.sh plan <root>                       Tokyo: plan.json of the 38 listed scenes (AlpaSim image) -> <root>/plan.json
#   rf_p4.sh job <root> <scene id | 8-char id> one scene (both phases + read); resumes (finished phases are skipped)
#   rf_p4.sh smoke <root>                      Tokyo: S1 (card 0) and S2 (card 1) in parallel, then S3; state in <root>/CHAIN_STATUS
#   rf_p4.sh fanout <root> <N | all>           box: one GPU-pool job per scene (python -m jevdrive.cl fanout) for the first N scenes of the
#                                              plan not yet submitted (staged launch: 1, then 10, then all; `all` adds the collect job)
#   rf_p4.sh read <root>                       merge the per-scene reads -> <root>/{read.json, table.md, summary.json}
# Env: BACKEND (tokyo | box; default: tokyo when /data/runs/route_night exists), CARD (Tokyo card, 1), TAG (RSG-L18-s0), PLAN (plan.json to
#   copy into a new root; default the committed experiments/route_follow/results/p4_self/plan.json), P4_D (distances, "40,30,20,10,0"),
#   P4_ONLY (cell suffixes, e.g. "g1.25-raw,g1-ve"), KEEP_ASL (1: keep rollout.asl, needed for rows; mp4s are always dropped).
# Serving = the reference /data/runs/route_night/r281/rsg: JEV_VCONT=1.0 JEV_LEAD=1, gains 20, holdlead off; plus
# send_recording_ground_truth=true (log fields, the log fallback) and PAI_DUMP (1 Hz model-input dumps for the rows' G-in gate).
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); repo=$(cd "$here/../../.." && pwd)
cmd=${1:?plan|job|smoke|fanout|read}; O=$(mkdir -p "${2:?root}" && cd "$2" && pwd); shift 2
D=${DATA_DIR:-/data}; T=${TAG:-RSG-L18-s0}; CARD=${CARD:-1}
BACKEND=${BACKEND:-$([[ -d /data/runs/route_night ]] && echo tokyo || echo box)}
WIZ=(runtime.simulation_config.send_recording_ground_truth=true controller.gains.rel_front_steering_angle_weight=20)
st() { echo "$(date '+%F %T') $*" | tee -a "$O/log.txt" | tee "$O/CHAIN_STATUS"; }
if [[ $BACKEND == tokyo ]]; then PY=python3; else PY=$repo/.venv/bin/python; fi
py() { $PY "$here/rf_p4.py" "$@"; }
apy() {  # rf_p4.py in the AlpaSim environment (scene loader, route generator, polars); paths under $D/runs are the same inside
  if [[ $BACKEND == tokyo ]]; then
    docker run --rm --entrypoint bash -v /data/datasets/nurec:/mnt/nre-data:ro -v /data/third_party/alpasim/src:/repo/src \
      -v /data/third_party/alpasim/plugins:/repo/plugins -v /data/runs:/data/runs -v "$repo:/jev:ro" alpasim-base:0.89.0 \
      -c "umask 0000 && cd /repo && uv run --no-sync python /jev/experiments/route_follow/scripts/rf_p4.py $*" 2> >(grep -v 'Creating\|it/s' >&2)
  else
    "$D/third_party/alpasim/.venv/bin/python" "$here/rf_p4.py" "$@" 2> >(grep -v 'Creating\|it/s' >&2)
  fi; }
[[ -f $O/plan.json || $cmd == plan ]] || cp "${PLAN:-$repo/experiments/route_follow/results/p4_self/plan.json}" "$O/plan.json"

stack() {  # <run dir> <cells json> <n rollouts> <scene list>: one simulator launch of every cell in the cells file
  local r=$1 cj=$2 n=$3 sl=$4 rc
  rm -rf "$r" "$r.out"; mkdir -p "$r"
  if [[ $BACKEND == tokyo ]]; then
    mkdir -p "$r/driver"; chmod 777 "$r/driver"; cp "$cj" "$r/driver/p4cells.json"
    local L=$repo/experiments/alpasim/lib
    for f in onp_log_driver.py onp_p4.py; do cmp -s "$repo/experiments/onpol_pai/lib/$f" "$L/$f" || cp "$repo/experiments/onpol_pai/lib/$f" "$L/$f"; done
    GPU=$CARD CONC=4 PAI_BASEPORT=$((6400 + 100 * CARD)) TAG=$T DRV_PY=onp_log_driver.py \
      DRV_ENV="-e JEV_VCONT=1.0 -e JEV_LEAD=1 -e PAI_DUMP=100000 -v $L/onp_p4.py:/app/jev-drive/experiments/alpasim/lib/onp_p4.py:ro -v $D/runs/op_parity/runs/$T:/app/data/runs/op_parity/runs/$T:ro" \
      bash "$repo/experiments/alpasim/scripts/pai_run.sh" "$r" "$sl" "${WIZ[@]}" "runtime.simulation_config.n_rollouts=$n" > "$r.out" 2>&1; rc=$?
  else
    mkdir -p "$r/sim/driver-logs"; cp "$cj" "$r/sim/driver-logs/p4cells.json"; ln -sfn sim/driver-logs "$r/driver"
    env SH30_TAG="$T" JEV_VCONT=1.0 JEV_LEAD=1 PAI_DUMP=100000 DRV_PY=../../onpol_pai/lib/onp_log_driver.py CONC=4 \
      bash "$repo/experiments/alpasim/scripts/pai_native.sh" "$r/sim" "$sl" "${WIZ[@]}" "runtime.simulation_config.n_rollouts=$n" > "$r.out" 2>&1; rc=$?
    rm -rf "$r/sim/nre-cache" "$r/sim/driver-tmp"
  fi
  find "$r/sim/rollouts" -name '*.mp4' -delete 2>/dev/null
  [[ ${KEEP_ASL:-1} == 1 ]] || find "$r/sim/rollouts" -name 'rollout.asl' -delete 2>/dev/null
  local got; got=$($PY -c "import json,sys;print(len(json.load(open(sys.argv[1]))['rollouts']))" "$r/sim/aggregate/results-summary.json" 2>/dev/null || echo 0)
  local want=$(( n * $(grep -c . "$sl") ))                  # n_rollouts is per scene
  echo "$(date '+%F %T') stack $(basename "$r"): rc $rc, $got / $want rollouts" >> "$O/log.txt"
  (( rc == 0 && got == want ))
}

job() {  # <scene>: phase A, reach, phase B, read
  local sc=$1 sd A B n t0
  [[ $sc == clipgt-* ]] || sc=$($PY -c "import json,sys;print(next(s for s in json.load(open(sys.argv[1]))['scenes'] if s[7:15]==sys.argv[2]))" "$O/plan.json" "$sc")
  sd=$O/$sc; A=$sd/${sc:7:8}-a; B=$sd/${sc:7:8}-b; mkdir -p "$sd"; echo "$sc" > "$sd/scene.txt"; rm -f "$sd"/{DONE,ERROR}
  fail() { echo "$*" > "$sd/ERROR"; echo "$(date '+%F %T') ${sc:7:8} ERROR $*" >> "$O/log.txt"; return 1; }
  t0=$(date +%s)
  if [[ ! -f $A/P4DONE ]]; then
    py cells --plan "$O/plan.json" --scene "$sc" --phase A --out "$sd/cells_A.json" > /dev/null || { fail "cells A"; return 1; }
    stack "$A" "$sd/cells_A.json" 1 "$sd/scene.txt" || { fail "phase A stack (see $A.out)"; return 1; }
    echo $(( $(date +%s) - t0 )) > "$A/P4DONE"
  fi
  apy reach --plan "$O/plan.json" --run "$A" --out "$sd/reach.json" > "$sd/reach.log" 2>&1 || { fail "reach (see reach.log)"; return 1; }
  py cells --plan "$O/plan.json" --scene "$sc" --phase B --reach "$sd/reach.json" ${P4_D:+--d "$P4_D"} ${P4_ONLY:+--only "$P4_ONLY"} \
    --out "$sd/cells_B.json" > "$sd/cells_B.log" || { fail "cells B"; return 1; }
  n=$($PY -c "import json,sys;print(sum(len(v) for v in json.load(open(sys.argv[1]))['cells'].values()))" "$sd/cells_B.json")
  if (( n > 0 )) && [[ ! -f $B/P4DONE ]]; then
    t0=$(date +%s)
    stack "$B" "$sd/cells_B.json" "$n" "$sd/scene.txt" || { fail "phase B stack (see $B.out)"; return 1; }
    echo $(( $(date +%s) - t0 )) > "$B/P4DONE"
  fi
  apy read --plan "$O/plan.json" --root "$O" --scenes "$sd/scene.txt" --out "$sd/read.json" > "$sd/read.log" 2>&1 || { fail "read (see read.log)"; return 1; }
  date > "$sd/DONE"; echo "$(date '+%F %T') ${sc:7:8} done: phase A $(cat "$A/P4DONE") s, phase B $n cells $(cat "$B/P4DONE" 2>/dev/null || echo 0) s" >> "$O/log.txt"
}

case $cmd in
plan)
  [[ $BACKEND == tokyo ]] || { echo "plan runs on Tokyo (the 38 scenes' reference runs are there)" >&2; exit 2; }
  head -38 /data/runs/route_night/pait/roll.txt > "$O/scenes38.txt"
  for f in "$repo"/experiments/alpasim/results/pai_exam/{hard_exam,control,exam_turn26}.txt /data/runs/route_night/exam_turn26.txt; do   # never the exam / control
    [[ -f $f ]] || continue
    n=$(grep -o 'clipgt-[0-9a-f-]*' "$f" | grep -cxFf - "$O/scenes38.txt"); (( n == 0 )) || { echo "$n scenes of $f in the P4 list" >&2; exit 1; }
  done
  apy plan --scenes "$O/scenes38.txt" --roots /data/runs/route_night/r281/rsg /data/runs/route_night/pait/pass --out "$O/plan.json" ;;
job)
  job "$1" ;;
smoke)
  [[ $BACKEND == tokyo ]] || { echo "smoke runs on Tokyo" >&2; exit 2; }
  rm -f "$O"/SMOKE_{DONE,FAIL}; S1SC=${S1SC:-a174faed eaba3f6a}; S2SC=${S2SC:-f9e361aa}
  st "P4 smoke, code $(git -C "$repo" rev-parse --short HEAD): S1 ($S1SC) on card 0, S2 ($S2SC D20) on card 1"
  (  # S1: each S1 scene twice (two identity sessions in one launch) against the stored reference
    mkdir -p "$O/s1"; : > "$O/s1/scenes.txt"
    for s in $S1SC; do $PY -c "import json,sys;print(next(s for s in json.load(open(sys.argv[1]))['scenes'] if s[7:15]==sys.argv[2]))" "$O/plan.json" "$s" >> "$O/s1/scenes.txt"; done
    $PY - "$O/plan.json" "$O/s1/scenes.txt" "$O/s1/cells.json" <<'PY'
import json, sys
P = json.load(open(sys.argv[1]))["scenes"]; sc = open(sys.argv[2]).read().split()
k = ("u", "x", "y", "kappa", "s_entry", "s_exit", "s_release", "v_e")
c = dict(tag="id", D=None, src="self", g=1.0, lon="raw", switch=0)
json.dump(dict(scenes={s: {x: P[s][x] for x in k} for s in sc}, cells={s: [c, c] for s in sc}), open(sys.argv[3], "w"))
PY
    CARD=0 stack "$O/s1/s1-id" "$O/s1/cells.json" 2 "$O/s1/scenes.txt" || { echo "S1 run" > "$O/SMOKE_FAIL"; exit 1; }
    py s1 --run "$O/s1/s1-id" --ref /data/runs/route_night/r281/rsg > "$O/s1.txt" 2>&1 ) &
  p1=$!; sleep 30
  ( CARD=1 P4_D=20 P4_ONLY=${S2ONLY:-g1.25-raw,g1-ve,g1.25-ve08} job "$S2SC" || { echo "S2 run" > "$O/SMOKE_FAIL"; exit 1; }
    sc=$(cat "$O"/clipgt-"$S2SC"*/scene.txt); sd=$O/$sc
    py s1 --run "$sd/${sc:7:8}-a" --ref /data/runs/route_night/r281/rsg >> "$O/s1.txt" 2>&1
    py s2 --plan "$O/plan.json" --run "$sd/${sc:7:8}-b" > "$O/s2.txt" 2>&1 ) &
  p2=$!; wait $p1 $p2
  [[ -f $O/SMOKE_FAIL ]] && { st "smoke run failed: $(cat "$O/SMOKE_FAIL")"; exit 1; }
  st "S1: $(grep '^S1:' "$O/s1.txt" | head -1)"
  py merge --root "$O" --out "$O/read.json" > /dev/null && py table --read "$O/read.json" --plan "$O/plan.json" --md "$O/table.md" > /dev/null
  st "S3: $(python3 -c "import json;d=json.load(open('$O/read.json'));print(len(d['rows']), 'rows read')")"; date > "$O/SMOKE_DONE" ;;
fanout)
  [[ $BACKEND == box ]] || { echo "fanout runs on the GPU box" >&2; exit 2; }
  N=${1:?N or all}; cd "$repo" || exit 1
  arms=$($PY - "$O" "$N" <<'PY'
import json, os, sys
O, N = sys.argv[1], sys.argv[2]
sc = [s for s in json.load(open(f"{O}/plan.json"))["scenes"] if not os.path.exists(f"{O}/{s}/SUBMITTED")]
print(",".join(s[7:15] for s in (sc if N == "all" else sc[: int(N)])))
PY
)
  [[ -n $arms ]] || { echo "nothing left to submit"; exit 0; }
  col=(); [[ $N == all ]] && col=(--collect "bash $here/rf_p4.sh read $O")
  .venv/bin/python -m jevdrive.cl fanout --name p4 --arms "$arms" --vram 20 --cpu 6 --ram 40 --timeout-h 4 --tries 2 \
    --env CL_HIST_KEY=alpasim-read4 --log-dir "$O/pool/{arm}" "${col[@]}" -- bash "$here/rf_p4.sh" job "$O" "{arm}" | tee -a "$O/jobs.txt" || exit 1
  for a in ${arms//,/ }; do s=$($PY -c "import json,sys;print(next(s for s in json.load(open(sys.argv[1]))['scenes'] if s[7:15]==sys.argv[2]))" "$O/plan.json" "$a")
    mkdir -p "$O/$s"; date > "$O/$s/SUBMITTED"; done ;;
read)
  py merge --root "$O" --out "$O/read.json" && py table --read "$O/read.json" --plan "$O/plan.json" --md "$O/table.md" --json "$O/summary.json" ;;
*) echo "unknown command $cmd" >&2; exit 2 ;;
esac
