#!/usr/bin/env bash
# ROUTE-MEM chain (plans/2026-10-12-route-mem-prereg.md). Every job goes through the GPU pool; ids are appended to $W/jobs.txt.
#   submit-build                 nav features of the 12 navtrain shards (one CPU job each) -> yaw-rate rows; the on1_L18YS pool rows
#   submit-train <seed..>        RM-L18-s<seed>: rm_train.py train (YS2G's recipe + the branch; reliance read inside the job)
#   submit-read <TAG> [after]    one checkpoint on held 460 at 18 decisions in 6 jobs (cards 1-5), then the coverage summary
#   submit-served <TAG>          the 5-scene served-vs-offline check (G-serve): readlist on lists/serve5.txt, then rm_build.py served
#   read <TAG> <i> <n> | readlist <TAG> <list> <dir>    job bodies (ol_fast.sh's sim: the same driver path and overrides)
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/route_mem; OL=$DATA_DIR/runs/onpol_long; W=$DATA_DIR/runs/route_mem; CK=$DATA_DIR/runs/op_parity/runs
C=$DATA_DIR/datasets/alpasim_nuplan_long/after160; KEY=runtime.scene_provider.trajdata
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python
mkdir -p $W/pool
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
LONG=("$KEY.cache_location=$C" "$KEY.dataset.extra_params.num_timesteps_after=160")
sub() { local name=$1 after=$2; shift 2; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner route-mem --name "$name" --tries 1 --priority ${PRIO:-14} --no-check \
          ${after:+--after $after} --log-dir "$W/pool/$name" "$@" | awk '{print $NF}') && [[ -n $id ]] || die "submit $name"; echo "$id"; echo "$id $name" >> $W/jobs.txt; }
sim() {  # = ol_fast.sh sim: one 18-decision run with the sh30 driver
  local T=$1 L=$2 D=$3 DUMP=${4:-0}
  [[ -f $CK/$T/ckpt-final.pt ]] || die "no checkpoint $T"; [[ ! -e $D/aggregate/results-summary.json ]] || die "$D already ran"
  rm -rf "$D"; mkdir -p "$D"
  env SH30_TAG=$T SH30_DUMP=$DUMP bash experiments/alpasim/scripts/run.sh "$D" sh30 --scene-list "$L" "${BASE[@]}" "${LONG[@]}" || die "run.sh rc $?"
  [[ -f $D/aggregate/results-summary.json ]] || die "no results-summary"; }
prune() {  # ol_fast.sh prune (read): drop rollout.asl of scored, unfailed rollouts
  python3 - "$1" <<'PY'
import glob, json, os, sys
D = sys.argv[1]; n = 0
for r in json.load(open(f"{D}/aggregate/results-summary.json"))["rollouts"]:
    if not r.get("failure_reason") and (r.get("score") or 0) > 0:
        for f in glob.glob(f"{D}/rollouts/{r['clipgt_id']}/*/rollout.asl"):
            os.remove(f); n += 1
print("pruned", n, "rollout.asl")
PY
}
STAGE=${1:?stage}; shift
case $STAGE in
read)      T=$1; i=$2; n=$3; D=$W/read/$T/c$i/run; sim "$T" "$OL/lists/held_n$n/c$i.txt" "$D"; prune "$D" ;;
readlist)  sim "$1" "$2" "$3"; prune "$3" ;;
submit-build)
  ids=(); D=()
  for i in $(seq 0 11); do d=navtrain_full.s${i}of12; D+=($d)
    ids+=($(sub rm-nav-$i "" --vram 0.5 --cpu 12 --ram 40 --timeout-h 1 -- $P $E/scripts/rm_build.py nav --data $d --workers 12)) || exit 1; done
  sub rm-yr "$(IFS=,; echo "${ids[*]}")" --vram 0.5 --cpu 2 --ram 30 --timeout-h 1 -- $P $E/scripts/rm_build.py yr --data "${D[@]}" || exit 1
  sub rm-pool "" --vram 0.5 --cpu 32 --ram 80 --timeout-h 1 -- $P $E/scripts/rm_build.py pool --name L18YS --workers 32 || exit 1 ;;
submit-train)   # after=$AFTER (comma ids) if set
  for s in "$@"; do
    sub rm-train-s$s "${AFTER:-}" --train --vram 28 --cpu 6 --ram 70 --timeout-h 3 -- $TR $E/scripts/rm_train.py train --tag YRM-L18-s$s --init YS-L18-s1 \
      --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --on on1_L18YS \
      --on-k 13 --on-split alpasim_nuplan/onpol-train --on-mode self --seed $s || exit 1
  done ;;
submit-read)
  T=$1; A=${2:-}; rr=()
  for i in 0 1 2 3 4 5; do
    rr+=($(sub rm-read-$T-$i "$A" --vram 32 --cpu 8 --ram 40 --timeout-h 2 --gpus 1,2,3,4,5 --pin-strict --env CL_HIST_KEY=ol-fast-read -- bash $E/scripts/rm_chain.sh read $T $i 6)) || exit 1
  done
  dd=$(for i in 0 1 2 3 4 5; do printf "%s," $W/read/$T/c$i/run; done)
  sub rm-cover-$T "$(IFS=,; echo "${rr[*]}")" --vram 0.5 --cpu 2 --ram 16 --timeout-h 1 -- $P $E/scripts/rm_build.py cover --runs "$T=${dd%,}" --out $W/cover/$T.parquet > /dev/null || exit 1 ;;
submit-served)
  T=$1; L=$W/lists/serve5.txt; [[ -f $L ]] || die "no $L"
  s=$(sub rm-serve-$T "${2:-}" --vram 24 --cpu 6 --ram 30 --timeout-h 1 --gpus 1,2,3,4,5 --pin-strict -- bash $E/scripts/rm_chain.sh readlist $T $L $W/served_runs/$T) || exit 1
  sub rm-served-chk-$T "$s" --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $P $E/scripts/rm_build.py served --run $W/served_runs/$T --tag $T --workers 4 > /dev/null || exit 1 ;;
submit-zero)    # RMI-L18 = YS-L18-s1 + the branch at zero; 5-scene reads of both on cards 1-5; served check and plan equality
  z=$(sub rm-zero "" --vram 8 --cpu 4 --ram 30 --timeout-h 1 -- $TR $E/scripts/rm_train.py zero --init YS-L18-s1 --tag RMI-L18) || exit 1
  L=$W/lists/serve5.txt; [[ -f $L ]] || die "no $L"
  b=$(sub rm-serve-YS-L18-s1 "" --vram 24 --cpu 6 --ram 30 --timeout-h 1 --gpus 1,2,3,4,5 --pin-strict -- bash $E/scripts/rm_chain.sh readlist YS-L18-s1 $L $W/served_runs/YS-L18-s1) || exit 1
  s=$(sub rm-serve-RMI-L18 "$z" --vram 24 --cpu 6 --ram 30 --timeout-h 1 --gpus 1,2,3,4,5 --pin-strict -- bash $E/scripts/rm_chain.sh readlist RMI-L18 $L $W/served_runs/RMI-L18) || exit 1
  sub rm-served-chk-RMI-L18 "$s" --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $P $E/scripts/rm_build.py served --run $W/served_runs/RMI-L18 --tag RMI-L18 --workers 4 > /dev/null || exit 1
  sub rm-same-RMI-L18 "$s,$b" --vram 0.5 --cpu 2 --ram 8 --timeout-h 1 -- $P $E/scripts/rm_build.py same --a $W/served_runs/RMI-L18 --b $W/served_runs/YS-L18-s1 --label RMI-L18 > /dev/null || exit 1 ;;
*) die "unknown stage $STAGE" ;;
esac
