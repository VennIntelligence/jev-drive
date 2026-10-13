#!/usr/bin/env bash
# ROUTE-ONLY reads (plans/2026-10-13-route-only-prereg.md): RSG-L18-s0 with the served path replaced by the carried route (JEV_ROUTE,
# experiments/alpasim/lib/route_path.py). Every job goes through the GPU pool; ids are appended to $W/jobs.txt. Box, repo root:
#   submit-smoke                 arms off / own / log on lists/smoke.txt (2 held turn scenes, 18 decisions)
#   submit-held <arm> [after]    held 460 at 18 decisions in 6 chunks (onpol_long held_n6 lists) -> $W/held/<arm>/c<i>/run
#   submit-pub <arm> [after]     public 1491 at 10 decisions in 10 chunks (onpol_long all_n10 lists) -> $W/pub10/<arm>/b<i>
#   submit-post <held|pub> <after ids>   the CPU report job after the reads (post-held / post-pub bodies)
#   post-held | post-pub         ol_confirm.py paired vs the stored RSG-L18-s0 read, ro_report.py, rf_follow.py (held) -> $W/report/
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
submit-post)
  sub ro-post-$1 "$2" --vram 0.5 --cpu 32 --ram 48 --timeout-h 2 -- bash $0 post-$1 | tail -1 ;;
post-held)
  P=$DATA_DIR/third_party/alpasim/.venv/bin/python; O=$W/report/held; mkdir -p $O
  R=("RSG-L18-s0@held18=$(ls -d $DATA_DIR/runs/route_mem/read/$T/c*/run | paste -sd,)"); Q=("RSG-L18-s0=${R[0]#*=}")
  for A in own log bridge; do L=R-$A; [[ $A == log ]] && L=R-log-priv
    ls $W/held/$A/c*/run/aggregate/results-summary.json > /dev/null 2>&1 || continue
    R+=("$L@held18=$(ls -d $W/held/$A/c*/run | paste -sd,)"); Q+=("$L=${R[-1]#*=}"); done
  $P experiments/onpol_long/scripts/ol_confirm.py --runs "${R[@]}" --base $T --pairs R-bridge:R-own --out $O/confirm; rc1=$?
  $P experiments/route_only/scripts/ro_report.py --runs "${Q[@]}" --out $O/summary; rc2=$?
  F=(); for q in "${Q[@]}"; do F+=("${q%%=*}@18=${q#*=}"); done
  have=" ${F[*]%%=*} "; PP=()
  for pr in R-own@18:RSG-L18-s0@18 R-bridge@18:RSG-L18-s0@18 R-log-priv@18:RSG-L18-s0@18 R-bridge@18:R-own@18; do
    [[ $have == *" ${pr%%:*}="* || $have == *"${pr%%:*}"* ]] && [[ $have == *"${pr#*:}"* ]] && PP+=($pr); done
  $P experiments/route_follow/scripts/rf_follow.py --runs "${F[@]}" --out $O/follow --workers 32 --pairs "${PP[@]}"; rc3=$?
  echo "confirm $rc1 summary $rc2 follow $rc3"; exit $((rc1 + rc2 + rc3)) ;;
post-pub)
  P=$DATA_DIR/third_party/alpasim/.venv/bin/python; O=$W/report/pub; mkdir -p $O; G=$DATA_DIR/runs/route_follow/guard/read10
  R=("RSG-L18-s0@all10=$(ls -d $G/$T/b* | paste -sd,)"); Q=("RSG-L18-s0=${R[0]#*=}")
  for A in own bridge; do ls $W/pub10/$A/b*/aggregate/results-summary.json > /dev/null 2>&1 || continue
    R+=("R-$A@all10=$(ls -d $W/pub10/$A/b* | paste -sd,)"); Q+=("R-$A=${R[-1]#*=}"); done
  $P experiments/onpol_long/scripts/ol_confirm.py --runs "${R[@]}" --base $T --out $O/confirm; rc1=$?
  $P experiments/route_only/scripts/ro_report.py --runs "${Q[@]}" --out $O/summary; rc2=$?
  echo "confirm $rc1 summary $rc2"; exit $((rc1 + rc2)) ;;
*) die "unknown stage $STAGE" ;;
esac
