#!/usr/bin/env bash
# ONPOL-LONG round 1 chain (plans/2026-10-12-ol1-prereg.md): on-policy rows at 18 decisions, hinge training, held 460 read at 18 decisions.
# Every stage is a GPU-pool job; the submit stages queue them with --after. Box, repo root:
#   bash experiments/onpol_long/scripts/ol_chain.sh submit-roll <BASE tag> <after ids,comma separated>   -> 6 rollout jobs (cl fanout)
#   bash experiments/onpol_long/scripts/ol_chain.sh submit-rest <BASE tag> <recipe Y|S> <rollout ids>    -> rows, training, read jobs
# Job bodies (run by the pool, not by hand): roll1 <BASE> <chunk>; geo <BASE>; tok <BASE>; strip <BASE>; read1 <tag>; report.
# Driver path = alpasim_long's 18-decision read: run.sh sh30, c0b_chain.OVERRIDES + the rebuilt cache after160 (num_timesteps_after 160).
# State: $DATA_DIR/runs/onpol_long/{roll/<BASE>/c<i>, rows/<name>, read/<tag>}; the pool's log dirs beside them.
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/onpol_long; ON=experiments/onpol_nuplan; SPL=$ON/results/split; W=$DATA_DIR/runs/onpol_long; CK=$DATA_DIR/runs/op_parity/runs
C=$DATA_DIR/datasets/alpasim_nuplan_long/after160; KEY=runtime.scene_provider.trajdata; NCH=6
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; NAV=$DATA_DIR/envs/navsim2/bin/python; TR=$DATA_DIR/envs/op-train/bin/python
CL() { .venv/bin/python -m jevdrive.cl "$@"; }
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
LONG=("$KEY.cache_location=$C" "$KEY.dataset.extra_params.num_timesteps_after=160")
lists() {  # the train split in NCH interleaved chunks; never a held scene
  local d=$W/lists; mkdir -p "$d"
  [[ -z $(comm -12 <(sort $SPL/train.txt) <(sort $SPL/held.txt)) ]] || die "train and held overlap"
  for i in $(seq 0 $((NCH - 1))); do awk -v i=$i -v n=$NCH '(NR - 1) % n == i' $SPL/train.txt > "$d/c$i.txt.tmp"
    if [[ -f $d/c$i.txt ]]; then cmp -s "$d/c$i.txt" "$d/c$i.txt.tmp" || die "$d/c$i.txt changed"; rm "$d/c$i.txt.tmp"; else mv "$d/c$i.txt.tmp" "$d/c$i.txt"; fi
  done; }
STAGE=${1:?stage}; shift
case $STAGE in
roll1)   # one chunk: rollouts with the frame dump, then the simulator state (onpol_nuplan's on_loop.py step), then ot2_loop's pruning
  T=$1; i=$2; D=$W/roll/$T/c$i/run; L=$W/lists/c$i.txt
  [[ -f $CK/$T/ckpt-final.pt ]] || die "no checkpoint $T"; [[ -f $C/nuplan_test/scenes_list.dill ]] || die "no cache $C"
  [[ -z $(comm -12 <(sort "$L") <(sort $SPL/held.txt)) ]] || die "$L has a held scene"
  [[ ! -e $D/aggregate/results-summary.json ]] || die "$D already ran"
  rm -rf "$D"; mkdir -p "$D"
  env SH30_TAG=$T SH30_DUMP=1000000 bash experiments/alpasim/scripts/run.sh "$D" sh30 --scene-list "$L" "${BASE[@]}" "${LONG[@]}" || die "run.sh rc $?"
  [[ -f $D/aggregate/results-summary.json ]] || die "no results-summary"
  $P experiments/alpasim/scripts/c1_extract.py logs --run "$D" --out "$D/onpol_sim.pkl" --jobs 8 || die "c1_extract rc $?"
  python3 - "$D" <<'PY'
import glob, json, os, sys
D = sys.argv[1]; n = 0
for r in json.load(open(f"{D}/aggregate/results-summary.json"))["rollouts"]:
    if not r.get("failure_reason"):
        for f in glob.glob(f"{D}/rollouts/{r['clipgt_id']}/*/rollout.asl"):
            os.remove(f); n += 1
print("pruned", n, "rollout.asl")
PY
  echo "$(date '+%F %T') roll1 $T c$i done: $(ls $D/driver-logs/dump | wc -l) dumps, $(du -sh $D | cut -f1)" ;;
submit-roll)
  T=$1; AFTER=${2:-}; lists
  [[ -f $CK/$T/ckpt-final.pt ]] || die "no checkpoint $T"
  mkdir -p "$W/roll/$T"
  CL fanout --arms "$(seq -s, 0 $((NCH - 1)))" --name "ol-roll-${T%%-F-s0}" --owner onpol-long --vram 32 --cpu 8 --ram 60 --tries 1 --priority ${PRIO:-13} \
     --timeout-h 3 --no-check ${AFTER:+--after $AFTER} --log-dir "$W/roll/$T/c{arm}/pool" -- bash $E/scripts/ol_chain.sh roll1 "$T" "{arm}" ;;
geo)     # rows of one base's rollouts (onpol_nuplan on_rows.py geo, unedited) + labels K and C (ol_labels.py build); CPU
  N=$2; RR=()
  for T in ${1//,/ }; do R=$W/roll/$T; RR+=("$R")
  python3 - "$R" "$T" "$NCH" <<'PY' || die "rollout chunks missing ($T)"
import json, os, sys
R, T, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = {f"{T}@c{i}": [f"{R}/c{i}/run"] for i in range(n) if os.path.exists(f"{R}/c{i}/run/onpol_sim.pkl")}
json.dump(m, open(f"{R}/manifest.json", "w"), indent=1)
print(len(m), "of", n, "chunks"); sys.exit(len(m) != n)
PY
  done
  $NAV $ON/scripts/on_rows.py geo --name "$N" --roll "${RR[@]}" --workers ${OL_WORKERS:-48} || die "on_rows geo rc $?"
  $NAV $E/scripts/ol_labels.py build --name "$N" --workers ${OL_WORKERS:-48} || die "ol_labels build rc $?" ;;
tok)     # tokens + teacher + gate R1 (on_rows.py tok, unedited), then the pool gates (ol_labels.py gate: R1 / R2 as amended, K0); GPU
  N=$1; CTL=$2
  $TR $ON/scripts/on_rows.py tok --name "$N" --control "$CTL" || die "on_rows tok rc $?"
  $TR $E/scripts/ol_labels.py gate --name "$N" --out "$W/rows/$N" || die "pool gates failed ($W/rows/$N/gates.md): the family stops here" ;;
strip)   # after the tokens: drop the slot frames (`cur`, 98 % of a dump) from this run's own dumps; JPEG, poses, ego, mu stay
  N=$2
  [[ -f $DATA_DIR/runs/op_parity/cache/on1_$N@warp/front.npy && -f $W/rows/$N/gates.json ]] || die "no tokens / gates for $N"
  for T in ${1//,/ }; do
  du -sm $W/roll/$T/c*/run/driver-logs/dump | awk '{s += $1} END {print "dumps before strip:", s, "MB"}'
  $TR - "$W/roll/$T" <<'PY' || die "strip rc $?"
import glob, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
def one(f):
    z = dict(np.load(f))
    if "cur" not in z:
        return 0
    del z["cur"]
    np.savez_compressed(f + ".tmp.npz", **z)
    os.replace(f + ".tmp.npz", f)
    return 1
fs = sorted(glob.glob(f"{sys.argv[1]}/c*/run/driver-logs/dump/*.npz"))
with ProcessPoolExecutor(16) as ex:
    print("stripped", sum(ex.map(one, fs, chunksize=64)), "of", len(fs))
PY
  du -sm $W/roll/$T/c*/run/driver-logs/dump | awk '{s += $1} END {print "dumps after strip:", s, "MB"}'
  done ;;
read1)   # one checkpoint on the held 460 scenes at 18 decisions (alpasim_long's driver path), then ot2_loop's pruning
  TAG=$1; D=$W/read/$TAG/run
  [[ -f $CK/$TAG/ckpt-final.pt ]] || die "no checkpoint $TAG"; [[ $(wc -l < $SPL/held.txt) -eq 460 ]] || die "held.txt"
  [[ ! -e $D/aggregate/results-summary.json ]] || die "$D already ran"
  rm -rf "$D"; mkdir -p "$D"
  env SH30_TAG=$TAG bash experiments/alpasim/scripts/run.sh "$D" sh30 --scene-list $SPL/held.txt "${BASE[@]}" "${LONG[@]}" || die "run.sh rc $?"
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
report)  # the registered read of one family against its control and base (ol_report.py); ARM = SKC (round 1) or S (Amendment 3)
  F=$1; A=${2:-SKC}; B=$( [[ $F == Y ]] && echo YR10m10-F || echo P2H10S-F ); H=$DATA_DIR/runs/alpasim_long/held/h18
  R=(); for t in ${F}$A-L18-s0 ${F}$A-L18-s1 ${F}NONE-L18-s0 ${F}NONE-L18-s1; do R+=("$t@18=$W/read/$t/run"); done
  for t in $B-s0 $B-s1; do [[ -f $H/$t/aggregate/results-summary.json ]] && R+=("$t@18=$H/$t"); done
  $P $E/scripts/ol_report.py --family $F --arm $A --runs "${R[@]}" --out $W/report/$F$([[ $A == SKC ]] || echo "-$A") || die "ol_report rc $?" ;;
submit-rest)   # everything after the rollouts of one base, chained with --after
  T=$1; F=$2; RIDS=$3; AL=${4:-}; N=L18$F; B=${T%-s0}; CTL=$( [[ $F == Y ]] && echo smoke10eY || echo smoke10e )
  [[ -f $DATA_DIR/runs/onpol_nuplan/rows/$CTL/r1.json ]] || die "no interpreted control $CTL"
  sub() { local name=$1 after=$2; shift 2; local id; id=$(CL submit --owner onpol-long --name "$name" --tries 1 --priority ${PRIO:-13} --no-check ${after:+--after $after} \
            --log-dir "$W/pool/$name" "$@" | awk '{print $NF}') && [[ -n $id ]] || die "submit $name"; echo "$id"; echo "$id $name" >> $W/jobs.txt; }
  g=$(sub ol-geo-$F "$RIDS" --vram 0.5 --cpu 48 --ram 96 --timeout-h 2 -- bash $E/scripts/ol_chain.sh geo $T $N) || exit 1
  k=$(sub ol-tok-$F "$g" --vram 16 --cpu 16 --ram 64 --timeout-h 2 -- bash $E/scripts/ol_chain.sh tok $N $CTL) || exit 1
  sub ol-strip-$F "$k" --vram 0.5 --cpu 16 --ram 16 --timeout-h 2 -- bash $E/scripts/ol_chain.sh strip $T $N > /dev/null || exit 1
  if [[ $F == Y ]]; then REC="--yr-mass 0.1 --label-bank"; SK="--agent-lam 10 --agent-on-only --shape --ho-w 3"
  else REC="--ho ot1:4,yr1:4,bd4:5 --ho-w 3 --ho-excl navsim/body1-val-logs --agent-lam 10 --shape --label-bank"; SK=""; fi
  ONR="--on on1_$N --on-k 13 --on-split alpasim_nuplan/onpol-train --on-mode self --on-lon ol_kc"
  OT="env PYTHONPATH=$PWD/$E/lib $TR $ON/scripts/on_train.py train --exp onpol_long --init $T --steps ${OL_STEPS:-10000}"
  rids=()
  for s in 0 1; do
    a=$(sub ol-train-${F}SKC-s$s "$k" --train --vram 26 --cpu 6 --ram 60 --timeout-h 4 -- $OT $REC $SK $ONR --seed $s --tag ${F}SKC-L18-s$s) || exit 1
    c=$(sub ol-train-${F}NONE-s$s "$AL" --train --vram 26 --cpu 6 --ram 60 --timeout-h 4 -- $OT $REC --seed $s --tag ${F}NONE-L18-s$s) || exit 1
    for x in "$a ${F}SKC-L18-s$s" "$c ${F}NONE-L18-s$s"; do set -- $x
      rids+=($(sub ol-read-$2 "$1" --vram 32 --cpu 8 --ram 90 --timeout-h 3 -- bash $E/scripts/ol_chain.sh read1 $2)) || exit 1; done
  done
  sub ol-report-$F "$(IFS=,; echo "${rids[*]}")" --vram 0.5 --cpu 16 --ram 32 --timeout-h 1 -- bash $E/scripts/ol_chain.sh report $F > /dev/null || exit 1
  echo "submitted for $T: $(tail -n 12 $W/jobs.txt | tr '\n' ';')" ;;
submit-skc)    # Amendment 2: the arm's half again (geo -> tok -> strip; tok -> arm trainings -> reads; all reads -> report), the ON-NONE reads already queued
  T=$1; F=$2; NONE_READS=$3; N=L18$F; CTL=$( [[ $F == Y ]] && echo smoke10eY || echo smoke10e )
  sub() { local name=$1 after=$2; shift 2; local id; id=$(CL submit --owner onpol-long --name "$name" --tries 1 --priority ${PRIO:-13} --no-check ${after:+--after $after} \
            --log-dir "$W/pool/$name-a2" "$@" | awk '{print $NF}') && [[ -n $id ]] || die "submit $name"; echo "$id"; echo "$id $name (amendment 2)" >> $W/jobs.txt; }
  g=$(sub ol-geo-$F "" --vram 0.5 --cpu 48 --ram 96 --timeout-h 2 -- bash $E/scripts/ol_chain.sh geo $T $N) || exit 1
  k=$(sub ol-tok-$F "$g" --vram 16 --cpu 16 --ram 64 --timeout-h 2 -- bash $E/scripts/ol_chain.sh tok $N $CTL) || exit 1
  sub ol-strip-$F "$k" --vram 0.5 --cpu 16 --ram 16 --timeout-h 2 -- bash $E/scripts/ol_chain.sh strip $T $N > /dev/null || exit 1
  if [[ $F == Y ]]; then REC="--yr-mass 0.1 --label-bank"; SK="--agent-lam 10 --agent-on-only --shape --ho-w 3"
  else REC="--ho ot1:4,yr1:4,bd4:5 --ho-w 3 --ho-excl navsim/body1-val-logs --agent-lam 10 --shape --label-bank"; SK=""; fi
  ONR="--on on1_$N --on-k 13 --on-split alpasim_nuplan/onpol-train --on-mode self --on-lon ol_kc"
  OT="env PYTHONPATH=$PWD/$E/lib $TR $ON/scripts/on_train.py train --exp onpol_long --init $T --steps ${OL_STEPS:-10000}"
  rids=($(echo $NONE_READS | tr , ' '))
  for s in 0 1; do
    a=$(sub ol-train-${F}SKC-s$s "$k" --train --vram 26 --cpu 6 --ram 60 --timeout-h 4 -- $OT $REC $SK $ONR --seed $s --tag ${F}SKC-L18-s$s) || exit 1
    r=$(sub ol-read-${F}SKC-L18-s$s "$a" --vram 32 --cpu 8 --ram 90 --timeout-h 3 -- bash $E/scripts/ol_chain.sh read1 ${F}SKC-L18-s$s) || exit 1; rids+=($r)
  done
  sub ol-report-$F "$(IFS=,; echo "${rids[*]}")" --vram 0.5 --cpu 16 --ram 32 --timeout-h 1 -- bash $E/scripts/ol_chain.sh report $F > /dev/null || exit 1
  echo "submitted for $T: $(tail -n 8 $W/jobs.txt | tr '\n' ';')" ;;
submit-s)      # Amendment 3: the S-only arm on the same pool (decision 259's recipe, no --on-lon), trainings -> reads -> report
  T=$1; F=$2; N=L18$F
  [[ -f $W/rows/$N/gates.json ]] && python3 -c "import json,sys; sys.exit(not json.load(open(sys.argv[1]))['pass'])" $W/rows/$N/gates.json || die "pool gates of $N not passed"
  sub() { local name=$1 after=$2; shift 2; local id; id=$(CL submit --owner onpol-long --name "$name" --tries 1 --priority ${PRIO:-14} --no-check ${after:+--after $after} \
            --log-dir "$W/pool/$name" "$@" | awk '{print $NF}') && [[ -n $id ]] || die "submit $name"; echo "$id"; echo "$id $name (amendment 3)" >> $W/jobs.txt; }
  if [[ $F == Y ]]; then REC="--yr-mass 0.1 --label-bank"; SK="--agent-lam 10 --agent-on-only --shape --ho-w 3"
  else REC="--ho ot1:4,yr1:4,bd4:5 --ho-w 3 --ho-excl navsim/body1-val-logs --agent-lam 10 --shape --label-bank"; SK=""; fi
  OT="$TR $ON/scripts/on_train.py train --exp onpol_long --init $T --steps ${OL_STEPS:-10000}"
  rids=()
  for s in 0 1; do
    a=$(sub ol-train-${F}S-s$s "" --train --vram 26 --cpu 6 --ram 60 --timeout-h 4 -- $OT $REC $SK --on on1_$N --on-k 13 --on-split alpasim_nuplan/onpol-train --on-mode self --seed $s --tag ${F}S-L18-s$s) || exit 1
    r=$(sub ol-read-${F}S-L18-s$s "$a" --vram 32 --cpu 8 --ram 90 --timeout-h 3 -- bash $E/scripts/ol_chain.sh read1 ${F}S-L18-s$s) || exit 1; rids+=($r)
  done
  sub ol-report-${F}S "$(IFS=,; echo "${rids[*]}")" --vram 0.5 --cpu 16 --ram 32 --timeout-h 1 -- bash $E/scripts/ol_chain.sh report $F S > /dev/null || exit 1
  echo "submitted for $T: $(tail -n 5 $W/jobs.txt | tr '\n' ';')" ;;
readlist)  # one checkpoint on a scene list at 18 or 10 decisions (Amendment 5 confirmation reads); prune as read1
  TAG=$1; L=$2; LEN=$3; D=$4
  [[ -f $CK/$TAG/ckpt-final.pt ]] || die "no checkpoint $TAG"; [[ ! -e $D/aggregate/results-summary.json ]] || die "$D already ran"
  X=(); [[ $LEN == 18 ]] && X=("${LONG[@]}")
  rm -rf "$D"; mkdir -p "$D"
  env SH30_TAG=$TAG bash experiments/alpasim/scripts/run.sh "$D" sh30 --scene-list "$L" "${BASE[@]}" "${X[@]}" || die "run.sh rc $?"
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
submit-r2)    # Amendment 5: round 2 on the Y family from YS-L18-s1. Rollouts of YS-L18-s0 / -s1 on train at 18 decisions with dumps (= the train-18
              # confirmation read), the 1491-scene 10-decision confirmation read, pool L18YS (S rows), YS2 / YS2NONE from YS-L18-s1, held reads, reports
  I=YS-L18-s1; N=L18YS
  sub() { local name=$1 after=$2; shift 2; local id; id=$(CL submit --owner onpol-long --name "$name" --tries 1 --priority ${PRIO:-14} --no-check ${after:+--after $after} \
            --log-dir "$W/pool/$name" "$@" | awk '{print $NF}') && [[ -n $id ]] || die "submit $name"; echo "$id"; echo "$id $name (amendment 5)" >> $W/jobs.txt; }
  lists; ALL=$W/lists/all1491.txt; sort -u $SPL/train.txt $SPL/held.txt > $ALL; [[ $(wc -l < $ALL) -eq 1491 ]] || die "all1491"
  for i in 0 1 2 3; do awk -v i=$i '(NR - 1) % 4 == i' $ALL > $W/lists/a$i.txt; done
  rr=(); cf=()
  for T in YS-L18-s0 YS-L18-s1; do mkdir -p $W/roll/$T
    for i in $(seq 0 $((NCH - 1))); do rr+=($(sub ol-roll-$T-$i "" --vram 32 --cpu 8 --ram 60 --timeout-h 3 -- bash $E/scripts/ol_chain.sh roll1 $T $i)) || exit 1; done
    for i in 0 1 2 3; do cf+=($(sub ol-a10-$T-$i "" --vram 32 --cpu 8 --ram 69 --timeout-h 3 -- bash $E/scripts/ol_chain.sh readlist $T $W/lists/a$i.txt 10 $W/read10/$T/a$i)) || exit 1; done
  done
  sub ol-confirm "$(IFS=,; echo "${rr[*]},${cf[*]}")" --vram 0.5 --cpu 16 --ram 32 --timeout-h 1 -- bash $E/scripts/ol_chain.sh confirm > /dev/null || exit 1
  g=$(sub ol-geo-YS "$(IFS=,; echo "${rr[*]}")" --vram 0.5 --cpu 48 --ram 96 --timeout-h 2 -- bash $E/scripts/ol_chain.sh geo YS-L18-s0,YS-L18-s1 $N) || exit 1
  k=$(sub ol-tok-YS "$g" --vram 16 --cpu 16 --ram 64 --timeout-h 2 -- bash $E/scripts/ol_chain.sh tok $N smoke10eY) || exit 1
  sub ol-strip-YS "$k" --vram 0.5 --cpu 16 --ram 16 --timeout-h 2 -- bash $E/scripts/ol_chain.sh strip YS-L18-s0,YS-L18-s1 $N > /dev/null || exit 1
  OT="$TR $ON/scripts/on_train.py train --exp onpol_long --init $I --steps ${OL_STEPS:-10000} --yr-mass 0.1 --label-bank"
  rids=()
  for s in 0 1; do
    a=$(sub ol-train-YS2-s$s "$k" --train --vram 26 --cpu 6 --ram 60 --timeout-h 4 -- $OT --agent-lam 10 --agent-on-only --shape --ho-w 3 --on on1_$N --on-k 13 \
          --on-split alpasim_nuplan/onpol-train --on-mode self --seed $s --tag YS2-L18-s$s) || exit 1
    c=$(sub ol-train-YS2NONE-s$s "" --train --vram 26 --cpu 6 --ram 60 --timeout-h 4 -- $OT --seed $s --tag YS2NONE-L18-s$s) || exit 1
    for x in "$a YS2-L18-s$s" "$c YS2NONE-L18-s$s"; do set -- $x
      rids+=($(sub ol-read-$2 "$1" --vram 32 --cpu 8 --ram 90 --timeout-h 3 -- bash $E/scripts/ol_chain.sh read1 $2)) || exit 1; done
  done
  sub ol-report-YS2 "$(IFS=,; echo "${rids[*]}")" --vram 0.5 --cpu 16 --ram 32 --timeout-h 1 -- bash $E/scripts/ol_chain.sh report2 > /dev/null || exit 1
  echo "submitted: $(grep -c "amendment 5" $W/jobs.txt) jobs" ;;
confirm)      # Amendment 5 confirmation reads against YR10m10-F-s0: train 1031 at 18 (the base's own round-1 rollouts) and all 1491 at 10 (stored)
  A=$DATA_DIR/runs/alpasim; R=()
  for T in YR10m10-F-s0 YS-L18-s0 YS-L18-s1; do R+=("$T@train18=$(ls -d $W/roll/$T/c*/run | paste -sd,)"); done
  R+=("YR10m10-F-s0@all10=$(python3 -c "import json; a = json.load(open('$A/ot3/a/manifest.json')); b = json.load(open('$A/cf1/a/manifest.json')); print(','.join(a['YR10m10-F-s0'] + a['YR10m10-F-s0@new'] + b['YR10m10-F-s0@fresh']))")")
  for T in YS-L18-s0 YS-L18-s1; do R+=("$T@all10=$(ls -d $W/read10/$T/a*/ | sed 's#/$##' | paste -sd,)"); done
  $P $E/scripts/ol_confirm.py --runs "${R[@]}" --base YR10m10-F-s0 --out $W/report/confirm || die "ol_confirm rc $?" ;;
report2)      # Amendment 5 round-2 reads: against the previous round (YS-L18-s1, the start) and against the base YR10m10-F-s0
  H=$DATA_DIR/runs/alpasim_long/held/h18; R=()
  for t in YS2-L18-s0 YS2-L18-s1 YS2NONE-L18-s0 YS2NONE-L18-s1 YS-L18-s1 YS-L18-s0; do R+=("$t@18=$W/read/$t/run"); done
  for t in YR10m10-F-s0 YR10m10-F-s1; do R+=("$t@18=$H/$t"); done
  $P $E/scripts/ol_report.py --family Y --arm S2 --ctl S2NONE --base-tags YS-L18-s1,YS-L18-s0 --runs "${R[@]}" --out $W/report/Y-S2 || die "ol_report (prev round) rc $?"
  $P $E/scripts/ol_report.py --family Y --arm S2 --ctl S2NONE --base-tags YR10m10-F-s0,YR10m10-F-s1 --runs "${R[@]}" --out $W/report/Y-S2-base || die "ol_report (base) rc $?" ;;
*) die "unknown stage $STAGE" ;;
esac
