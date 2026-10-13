#!/usr/bin/env bash
# ROUTE-ONLY reads (plans/2026-10-13-route-only-prereg.md): RSG-L18-s0 with the served path replaced by the carried route (JEV_ROUTE,
# experiments/alpasim/lib/route_path.py). Every job goes through the GPU pool; ids are appended to $W/jobs.txt. Box, repo root:
#   submit-smoke                 arms off / own / log on lists/smoke.txt (2 held turn scenes, 18 decisions)
#   submit-held <arm> [after]    held 460 at 18 decisions in 6 chunks (onpol_long held_n6 lists) -> $W/held/<arm>/c<i>/run
#   submit-pub <arm> [after]     public 1491 at 10 decisions in 10 chunks (onpol_long all_n10 lists) -> $W/pub10/<arm>/b<i>
#   run <arm> <list> <dir> <len> job body (arm: off | own | bridge | log; len 18 = the after160 cache, 10 = the shipped one)
set -uo pipefail
cd "$(dirname "$0")/../../.."
T=RSG-L18-s0; OL=$DATA_DIR/runs/onpol_long; W=$DATA_DIR/runs/route_only; CK=$DATA_DIR/runs/op_parity/runs
C=$DATA_DIR/datasets/alpasim_nuplan_long/after160; KEY=runtime.scene_provider.trajdata
mkdir -p $W/pool
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
LONG=("$KEY.cache_location=$C" "$KEY.dataset.extra_params.num_timesteps_after=160")
sub() { local name=$1 after=$2; shift 2; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner route-only --name "$name" --tries 1 --priority ${PRIO:-12} --no-check \
          ${after:+--after $after} --log-dir "$W/pool/$name" "$@" | awk '{print $NF}') && [[ -n $id ]] || die "submit $name"; echo "$id"; echo "$id $name" >> $W/jobs.txt; }
RD=(--vram 32 --cpu 8 --ram 50 --timeout-h 2)
STAGE=${1:?stage}; shift
case $STAGE in
run)
  A=$1 L=$2 D=$3 N=$4; X=(); E=()
  [[ -f $CK/$T/ckpt-final.pt ]] || die "no checkpoint $T"; [[ ! -e $D/aggregate/results-summary.json ]] || die "$D already ran"
  [[ $N == 18 ]] && X=("${LONG[@]}")
  case $A in off) ;; own) E=(JEV_ROUTE=own) ;; bridge) E=(JEV_ROUTE=bridge) ;; log) E=(JEV_ROUTE=log); X+=(runtime.simulation_config.send_recording_ground_truth=true) ;; *) die "arm $A" ;; esac
  rm -rf "$D"; mkdir -p "$D"
  env SH30_TAG=$T "${E[@]}" bash experiments/alpasim/scripts/run.sh "$D" sh30 --scene-list "$L" "${BASE[@]}" "${X[@]}" || die "run.sh rc $?"
  [[ -f $D/aggregate/results-summary.json ]] || die "no results-summary"
  python3 - "$D" <<'PY'
import glob, json, os, sys
D = sys.argv[1]; n = 0
for r in json.load(open(f"{D}/aggregate/results-summary.json"))["rollouts"]:
    if not r.get("failure_reason") and (r.get("score") or 0) > 0:
        for f in glob.glob(f"{D}/rollouts/{r['clipgt_id']}/*/rollout.asl"):
            os.remove(f); n += 1
print("pruned", n, "rollout.asl")
PY
  ;;
submit-smoke)
  for A in ${ARMS:-off own log}; do sub ro-smoke-$A "" "${RD[@]}" -- bash $0 run $A $W/lists/smoke.txt $W/smoke/$A 18 > /dev/null || exit 1; done; tail -n 3 $W/jobs.txt ;;
submit-held)
  A=$1; for i in 0 1 2 3 4 5; do sub ro-held-$A-$i "${2:-}" "${RD[@]}" -- bash $0 run $A $OL/lists/held_n6/c$i.txt $W/held/$A/c$i/run 18 > /dev/null || exit 1; done; tail -n 6 $W/jobs.txt ;;
submit-pub)
  A=$1; for i in $(seq 0 9); do sub ro-pub-$A-$i "${2:-}" "${RD[@]}" -- bash $0 run $A $OL/lists/all_n10/c$i.txt $W/pub10/$A/b$i 10 > /dev/null || exit 1; done; tail -n 10 $W/jobs.txt ;;
*) die "unknown stage $STAGE" ;;
esac
