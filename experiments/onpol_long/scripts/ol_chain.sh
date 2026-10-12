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
submit-gentle)  # Amendment 6: round 1b, gentle continuation of YR10m10-F-s0 (lr x 0.1, warm-up 1 step, 2 500 steps), ON-NONE-gentle and S-only-gentle
  N=L18Y
  sub() { local name=$1 after=$2; shift 2; local id; id=$(CL submit --owner onpol-long --name "$name" --tries 1 --priority ${PRIO:-14} --no-check ${after:+--after $after} \
            --log-dir "$W/pool/$name" "$@" | awk '{print $NF}') && [[ -n $id ]] || die "submit $name"; echo "$id"; echo "$id $name (amendment 6)" >> $W/jobs.txt; }
  OT="$TR $ON/scripts/on_train.py train --exp onpol_long --init YR10m10-F-s0 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank"
  rids=()
  for s in 0 1; do
    a=$(sub ol-train-YSG-s$s "" --train --vram 26 --cpu 6 --ram 60 --timeout-h 2 -- $OT --agent-lam 10 --agent-on-only --shape --ho-w 3 --on on1_$N --on-k 13 \
          --on-split alpasim_nuplan/onpol-train --on-mode self --seed $s --tag YSG-L18-s$s) || exit 1
    c=$(sub ol-train-YNONEG-s$s "" --train --vram 26 --cpu 6 --ram 60 --timeout-h 2 -- $OT --seed $s --tag YNONEG-L18-s$s) || exit 1
    for x in "$a YSG-L18-s$s" "$c YNONEG-L18-s$s"; do set -- $x
      rids+=($(sub ol-read-$2 "$1" --vram 32 --cpu 8 --ram 90 --timeout-h 3 -- bash $E/scripts/ol_chain.sh read1 $2)) || exit 1; done
  done
  sub ol-report-YSG "$(IFS=,; echo "${rids[*]}")" --vram 0.5 --cpu 16 --ram 32 --timeout-h 1 -- bash $E/scripts/ol_chain.sh report-gentle > /dev/null || exit 1
  echo "submitted: $(grep -c "amendment 6" $W/jobs.txt) jobs" ;;
report-gentle)
  H=$DATA_DIR/runs/alpasim_long/held/h18; R=()
  for t in YSG-L18-s0 YSG-L18-s1 YNONEG-L18-s0 YNONEG-L18-s1 YS-L18-s0 YS-L18-s1; do R+=("$t@18=$W/read/$t/run"); done
  for t in YR10m10-F-s0 YR10m10-F-s1; do R+=("$t@18=$H/$t"); done
  $P $E/scripts/ol_report.py --family Y --arm SG --ctl NONEG --runs "${R[@]}" --out $W/report/Y-SG-base || die "ol_report (base) rc $?"
  $P $E/scripts/ol_report.py --family Y --arm SG --ctl NONEG --base-tags YS-L18-s0,YS-L18-s1 --ref-mean --runs "${R[@]}" --out $W/report/Y-SG-ys || die "ol_report (YS) rc $?" ;;
submit-r2g)    # Amendment 8: round 2b, round-2 rows (pool L18YS), gentle schedule (lr x 0.1, warm-up 1, 2 500 steps) from YS-L18-s1; reads off card 0
  I=YS-L18-s1; N=L18YS
  sub() { local name=$1 after=$2; shift 2; local id; id=$(CL submit --owner onpol-long --name "$name" --tries 1 --priority ${PRIO:-14} --no-check ${after:+--after $after} \
            --log-dir "$W/pool/$name" "$@" | awk '{print $NF}') && [[ -n $id ]] || die "submit $name"; echo "$id"; echo "$id $name (amendment 8)" >> $W/jobs.txt; }
  OT="$TR $ON/scripts/on_train.py train --exp onpol_long --init $I --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank"
  rids=()
  for s in 0 1; do
    a=$(sub ol-train-YS2G-s$s "" --train --vram 26 --cpu 6 --ram 60 --timeout-h 2 -- $OT --agent-lam 10 --agent-on-only --shape --ho-w 3 --on on1_$N --on-k 13 \
          --on-split alpasim_nuplan/onpol-train --on-mode self --seed $s --tag YS2G-L18-s$s) || exit 1
    c=$(sub ol-train-YS2GNONE-s$s "" --train --vram 26 --cpu 6 --ram 60 --timeout-h 2 -- $OT --seed $s --tag YS2GNONE-L18-s$s) || exit 1
    for x in "$a YS2G-L18-s$s" "$c YS2GNONE-L18-s$s"; do set -- $x
      rids+=($(sub ol-read-$2 "$1" --vram 32 --cpu 8 --ram 90 --timeout-h 3 --gpus 1,2,3,4,5 --pin-strict -- bash $E/scripts/ol_chain.sh read1 $2)) || exit 1; done
  done
  sub ol-report-YS2G "$(IFS=,; echo "${rids[*]}")" --vram 0.5 --cpu 16 --ram 32 --timeout-h 1 -- bash $E/scripts/ol_chain.sh report-r2g > /dev/null || exit 1
  echo "submitted: $(grep -c "amendment 8" $W/jobs.txt) jobs" ;;
report-r2g)
  H=$DATA_DIR/runs/alpasim_long/held/h18; R=()
  for t in YS2G-L18-s0 YS2G-L18-s1 YS2GNONE-L18-s0 YS2GNONE-L18-s1 YS-L18-s1 YS-L18-s0; do R+=("$t@18=$W/read/$t/run"); done
  for t in YR10m10-F-s0 YR10m10-F-s1; do R+=("$t@18=$H/$t"); done
  $P $E/scripts/ol_report.py --family Y --arm S2G --ctl S2GNONE --base-tags YS-L18-s1,YS-L18-s0 --runs "${R[@]}" --out $W/report/Y-S2G || die "ol_report (prev round) rc $?"
  $P $E/scripts/ol_report.py --family Y --arm S2G --ctl S2GNONE --base-tags YR10m10-F-s0,YR10m10-F-s1 --runs "${R[@]}" --out $W/report/Y-S2G-base || die "ol_report (base) rc $?" ;;
merge)        # Amendment 9: pool on1_L18YA = round-1 rows (on1_L18Y) + round-2 rows (on1_L18YS)
  $TR $E/scripts/ol_pool_merge.py --out on1_L18YA --pools on1_L18Y:r1 on1_L18YS:r2 || die "merge rc $?" ;;
submit-agg)    # Amendment 9: YS-AGG from YS-L18-s1 on the aggregated pool, gentle (YSAGG) and full schedule (YSAGGF), seeds 0 / 1; reads off card 0
  I=YS-L18-s1; N=L18YA
  sub() { local name=$1 after=$2; shift 2; local id; id=$(CL submit --owner onpol-long --name "$name" --tries 1 --priority ${PRIO:-14} --no-check ${after:+--after $after} \
            --log-dir "$W/pool/$name" "$@" | awk '{print $NF}') && [[ -n $id ]] || die "submit $name"; echo "$id"; echo "$id $name (amendment 9)" >> $W/jobs.txt; }
  m=$(sub ol-merge-YA "" --vram 0.5 --cpu 8 --ram 64 --timeout-h 1 -- bash $E/scripts/ol_chain.sh merge) || exit 1
  G="--steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500"; FU="--steps 10000"
  rids=()
  for v in G F; do sch=$([[ $v == G ]] && echo "$G" || echo "$FU"); tg=YSAGG$([[ $v == F ]] && echo F)
    for s in 0 1; do
      a=$(sub ol-train-$tg-s$s "$m" --train --vram 26 --cpu 6 --ram 60 --timeout-h 4 -- $TR $ON/scripts/on_train.py train --exp onpol_long --init $I $sch --yr-mass 0.1 \
            --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --on on1_$N --on-k 13 --on-split alpasim_nuplan/onpol-train --on-mode self --seed $s --tag $tg-L18-s$s) || exit 1
      rids+=($(sub ol-read-$tg-L18-s$s "$a" --vram 32 --cpu 8 --ram 90 --timeout-h 3 --gpus 1,2,3,4,5 --pin-strict -- bash $E/scripts/ol_chain.sh read1 $tg-L18-s$s)) || exit 1
    done
  done
  sub ol-report-YSAGG "$(IFS=,; echo "${rids[*]}"),1012-140808-4a0c" --vram 0.5 --cpu 16 --ram 32 --timeout-h 1 -- bash $E/scripts/ol_chain.sh report-agg > /dev/null || exit 1
  echo "submitted: $(grep -c "amendment 9" $W/jobs.txt) jobs" ;;
report-agg)    # each aggregated arm against YS-L18-s1 with round 2b's gentle control (gentle arm) / round 2's full control (full arm)
  H=$DATA_DIR/runs/alpasim_long/held/h18
  for v in G F; do tg=YSAGG$([[ $v == F ]] && echo F); c=$([[ $v == G ]] && echo S2GNONE || echo S2NONE); R=()
    for t in $tg-L18-s0 $tg-L18-s1 Y$c-L18-s0 Y$c-L18-s1 YS-L18-s1 YS-L18-s0; do R+=("$t@18=$W/read/$t/run"); done
    $P $E/scripts/ol_report.py --family Y --arm SAGG$([[ $v == F ]] && echo F) --ctl $c --base-tags YS-L18-s1,YS-L18-s0 --runs "${R[@]}" --out $W/report/Y-$tg || die "ol_report $tg rc $?"
  done ;;
submit-g10)    # Amendment 10: last-slot guardrail, YS2G-L18-s0 / -s1 on all 1491 public scenes at 10 decisions in 10 chunks each, off card 0; then confirm-g10
  sub() { local name=$1 after=$2; shift 2; local id; id=$(CL submit --owner onpol-long --name "$name" --tries 1 --priority ${PRIO:-14} --no-check ${after:+--after $after} \
            --log-dir "$W/pool/$name" "$@" | awk '{print $NF}') && [[ -n $id ]] || die "submit $name"; echo "$id"; echo "$id $name (amendment 10)" >> $W/jobs.txt; }
  ALL=$W/lists/all1491.txt; [[ $(wc -l < $ALL) -eq 1491 ]] || die "all1491"; mkdir -p $W/lists/all_n10
  for i in $(seq 0 9); do awk -v i=$i '(NR - 1) % 10 == i' $ALL > $W/lists/all_n10/c$i.txt; done
  cf=()
  for T in YS2G-L18-s0 YS2G-L18-s1; do
    for i in $(seq 0 9); do cf+=($(sub ol-a10-$T-$i "" --vram 32 --cpu 8 --ram 50 --timeout-h 2 --gpus 1,2,3,4,5 --pin-strict -- \
          bash $E/scripts/ol_chain.sh readlist $T $W/lists/all_n10/c$i.txt 10 $W/read10/$T/b$i)) || exit 1; done
  done
  sub ol-confirm-g10 "$(IFS=,; echo "${cf[*]}")" --vram 0.5 --cpu 16 --ram 32 --timeout-h 1 -- bash $E/scripts/ol_chain.sh confirm-g10 > /dev/null || exit 1
  echo "submitted: $(grep -c "amendment 10" $W/jobs.txt) jobs" ;;
confirm-g10)  # Amendment 10: YS2G and YS-L18 against YR10m10-F-s0 (stored 2026-10-09 rollouts) on all 1491 at 10 decisions; YS2G against YS-L18
  A=$DATA_DIR/runs/alpasim; R=()
  R+=("YR10m10-F-s0@all10=$(python3 -c "import json; a = json.load(open('$A/ot3/a/manifest.json')); b = json.load(open('$A/cf1/a/manifest.json')); print(','.join(a['YR10m10-F-s0'] + a['YR10m10-F-s0@new'] + b['YR10m10-F-s0@fresh']))")")
  for T in YS-L18-s0 YS-L18-s1 YS2G-L18-s0 YS2G-L18-s1; do R+=("$T@all10=$(ls -d $W/read10/$T/[ab]*/ | sed 's#/$##' | paste -sd,)"); done
  $P $E/scripts/ol_confirm.py --runs "${R[@]}" --base YR10m10-F-s0 --mean YS-L18=YS-L18-s0,YS-L18-s1 YS2G-L18=YS2G-L18-s0,YS2G-L18-s1 \
     --pairs YS2G-L18:YS-L18 YS2G-L18-s0:YS-L18-s1 YS2G-L18-s1:YS-L18-s1 --out $W/report/confirm-g10 || die "ol_confirm rc $?" ;;
submit-ego)   # Amendment 11: ego-status dropout on YS-L18's recipe (round-1 rows, YR10m10-F-s0, full schedule): YSED (p 0.5) and YSEX (p 1, removed also
              # when served), seeds 0 / 1; held reads at 18 decisions in 6 chunks off card 0; re-reads of YS-L18-s0 and YR10m10-F-s0 off card 0
              # (their stored held reads ran on card 0); open-loop reliance of every checkpoint; reports
  N=L18Y; F="env OL_RCH=6 bash $E/scripts/ol_fast.sh"
  sub() { local name=$1 after=$2; shift 2; local id; id=$(CL submit --owner onpol-long --name "$name" --tries 1 --priority ${PRIO:-14} --no-check ${after:+--after $after} \
            --log-dir "$W/pool/$name" "$@" | awk '{print $NF}') && [[ -n $id ]] || die "submit $name"; echo "$id"; echo "$id $name (amendment 11)" >> $W/jobs.txt; }
  RD="--vram 32 --cpu 8 --ram 40 --timeout-h 2 --gpus 1,2,3,4,5 --pin-strict --env CL_HIST_KEY=ol-fast-read"
  OT="$TR $ON/scripts/on_train.py train --exp onpol_long --init YR10m10-F-s0 --steps 10000 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 \
      --on on1_$N --on-k 13 --on-split alpasim_nuplan/onpol-train --on-mode self"
  rids=(); tids=()
  for x in "SED 0.5" "SEX 1.0"; do set -- $x
    for s in 0 1; do
      a=$(sub ol-train-Y$1-s$s "" --train --vram 26 --cpu 6 --ram 60 --timeout-h 4 -- $OT --ego-drop $2 --seed $s --tag Y$1-L18-s$s) || exit 1; tids+=($a)
      for i in $(seq 0 5); do rids+=($(sub ol-read-Y$1-L18-s$s-$i "$a" $RD -- $F read Y$1-L18-s$s $i 6)) || exit 1; done
    done
  done
  for T in YS-L18-s0 YR10m10-F-s0; do
    for i in $(seq 0 5); do rids+=($(sub ol-reread-$T-$i "" $RD -- $F readlist $T $W/lists/held_n6/c$i.txt $W/read/c15-$T/c$i/run)) || exit 1; done
  done
  r=$(sub ol-egorel "$(IFS=,; echo "${tids[*]}")" --vram 24 --cpu 6 --ram 60 --timeout-h 1 -- $TR $E/scripts/ol_ego.py rel --out $W/report/Y-ego/reliance.json \
        --tags YR10m10-F-s0 YR10m10-F-s1 YS-L18-s0 YS-L18-s1 YSED-L18-s0 YSED-L18-s1 YSEX-L18-s0 YSEX-L18-s1) || exit 1
  sub ol-report-ego "$(IFS=,; echo "${rids[*]}"),$r" --vram 0.5 --cpu 16 --ram 32 --timeout-h 1 -- bash $E/scripts/ol_chain.sh report-ego > /dev/null || exit 1
  echo "submitted: $(grep -c "amendment 11" $W/jobs.txt) jobs" ;;
report-ego)   # Amendment 11: each dropout arm against YS-L18 (control) and the base, all reads off card 0 except YR10m10-F-s1's (alpasim_long); drift / reliance
  H=$DATA_DIR/runs/alpasim_long/held/h18; c() { ls -d $W/read/$1/c*/run | paste -sd,; }
  R=("YS-L18-s0@18=$(c c15-YS-L18-s0)" "YS-L18-s1@18=$W/read/YS-L18-s1/run" "YR10m10-F-s0@18=$(c c15-YR10m10-F-s0)" "YR10m10-F-s1@18=$H/YR10m10-F-s1")
  for t in YSED-L18-s0 YSED-L18-s1 YSEX-L18-s0 YSEX-L18-s1; do R+=("$t@18=$(c $t)"); done
  for A in SED SEX; do
    $P $E/scripts/ol_report.py --family Y --arm $A --ctl S --base-tags YR10m10-F-s0,YR10m10-F-s1 --runs "${R[@]}" --out $W/report/Y-$A || die "ol_report $A rc $?"; done
  $P $E/scripts/ol_ego.py drift --reports $W/report/Y-SED $W/report/Y-SEX --rel $W/report/Y-ego/reliance.json --out $W/report/Y-ego \
     --tags YR10m10-F-s0 YR10m10-F-s1 YS-L18-s0 YS-L18-s1 YSED-L18-s0 YSED-L18-s1 YSEX-L18-s0 YSEX-L18-s1 \
     --groups YR=YR10m10-F-s0,YR10m10-F-s1 YS=YS-L18-s0,YS-L18-s1 YSED=YSED-L18-s0,YSED-L18-s1 YSEX=YSEX-L18-s0,YSEX-L18-s1 \
     --pairs YSED:YS YSEX:YS YS:YR YSED:YR YSEX:YR || die "ol_ego drift rc $?" ;;
submit-scorer)  # Amendment 12: candidate-scoring feasibility on YS-L18-s1's 43 zero events (ol_scorer.py prep -> replay -> geo -> score), one card off card 0
  sub() { local name=$1 after=$2; shift 2; local id; id=$(CL submit --owner onpol-long --name "$name" --tries 1 --priority ${PRIO:-15} --no-check ${after:+--after $after} \
            --log-dir "$W/pool/$name" "$@" | awk '{print $NF}') && [[ -n $id ]] || die "submit $name"; echo "$id"; echo "$id $name (amendment 12)" >> $W/jobs.txt; }
  S="$E/scripts/ol_scorer.py"; mkdir -p $W/scorer
  p=$(sub ol-sc-prep "" --vram 0.5 --cpu 16 --ram 32 --timeout-h 1 -- $P $S prep) || exit 1
  r=$(sub ol-sc-replay "$p" --vram 16 --cpu 8 --ram 40 --timeout-h 1 --gpus 1,2,3,4,5 --pin-strict -- env ALPASIM_SRC=$DATA_DIR/third_party/alpasim OMP_NUM_THREADS=2 $TR $S replay) || exit 1
  g=$(sub ol-sc-geo "$r" --vram 0.5 --cpu 16 --ram 48 --timeout-h 1 -- $NAV $S geo --workers 16) || exit 1
  sub ol-sc-score "$g" --vram 16 --cpu 8 --ram 48 --timeout-h 1 --gpus 1,2,3,4,5 -- $TR $S score > /dev/null || exit 1
  echo "submitted: $(grep -c "amendment 12" $W/jobs.txt) jobs" ;;
submit-armb)   # Amendment 13: arm B, directional road raster on pool on1_L18YS (CPU) -> YSB-L18-s{0,1} (round 2b's recipe with --on-road) -> held reads in
               # 6 chunks (ol_fast.sh read), all off card 0 -> report-armb (ol_report against YS2G and YS-L18, zm_read collect on merged views)
  sub() { local name=$1 after=$2; shift 2; local id; id=$(CL submit --owner onpol-long --name "$name" --tries 1 --priority ${PRIO:-14} --no-check ${after:+--after $after} \
            --log-dir "$W/pool/$name" "$@" | awk '{print $NF}') && [[ -n $id ]] || die "submit $name"; echo "$id"; echo "$id $name (amendment 13)" >> $W/jobs.txt; }
  RB=$W/rows/L18YS-dir; G15="--gpus 1,2,3,4,5 --pin-strict"; X1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
  d=$(sub ol-armb-dir "" --vram 0.5 --cpu 64 --ram 120 --timeout-h 2 -- $X1 $NAV $E/scripts/ol_dirraster.py build --out $RB --workers 64) || exit 1
  sub ol-armb-fig "" --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $X1 $NAV $E/scripts/ol_dirraster.py fig --events experiments/zeros_mech/results/events_YS-L18-s1.csv \
      --run $W/scorer/YS-L18-s1/run --out $RB/fig_turns.pkl > /dev/null || exit 1
  OT="$TR $ON/scripts/on_train.py train --exp onpol_long --init YS-L18-s1 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank"
  rids=()
  for s in 0 1; do
    a=$(sub ol-train-YSB-s$s "$d" --train --vram 26 --cpu 6 --ram 60 --timeout-h 2 $G15 -- $OT --agent-lam 10 --agent-on-only --shape --ho-w 3 --on on1_L18YS --on-k 13 \
          --on-split alpasim_nuplan/onpol-train --on-mode self --on-road $RB/road.npz --seed $s --tag YSB-L18-s$s) || exit 1
    for i in $(seq 0 5); do rids+=($(sub ol-read-YSB-L18-s$s-$i "$a" --vram 32 --cpu 8 --ram 40 --timeout-h 2 $G15 --env CL_HIST_KEY=ol-fast-read -- \
          env OL_RCH=6 bash $E/scripts/ol_fast.sh read YSB-L18-s$s $i 6)) || exit 1; done
  done
  sub ol-report-armb "$(IFS=,; echo "${rids[*]}")" --vram 0.5 --cpu 64 --ram 64 --timeout-h 1 -- bash $E/scripts/ol_chain.sh report-armb > /dev/null || exit 1
  echo "submitted: $(grep -c "amendment 13" $W/jobs.txt) jobs" ;;
report-armb)   # Amendment 13: YSB against YS2G (control) and YS-L18 (seed means); zm_read collect + report (unedited) on merged views; the registered lines
  H=$DATA_DIR/runs/alpasim_long/held/h18; c() { ls -d $W/read/$1/c*/run | paste -sd,; }
  R=("YSB-L18-s0@18=$(c YSB-L18-s0)" "YSB-L18-s1@18=$(c YSB-L18-s1)" "YS2G-L18-s0@18=$W/read/YS2G-L18-s0/run" "YS2G-L18-s1@18=$W/read/YS2G-L18-s1/run"
     "YS-L18-s0@18=$(c c15-YS-L18-s0)" "YS-L18-s1@18=$W/read/YS-L18-s1/run")
  $P $E/scripts/ol_report.py --family Y --arm SB --ctl S2G --base-tags YS-L18-s0,YS-L18-s1 --ref-mean --runs "${R[@]}" --out $W/report/Y-SB || die "ol_report rc $?"
  for t in YSB-L18-s0 YSB-L18-s1 c15-YS-L18-s0; do V=$W/read/$t/merged; rm -rf $V; mkdir -p $V/aggregate $V/driver-logs $V/rollouts
    cp $W/read/$t/c0/run/generated-user-config-0.yaml $V/; cat $W/read/$t/c*/run/driver-logs/drive.jsonl > $V/driver-logs/drive.jsonl
    for q in $W/read/$t/c*/run/rollouts/*; do ln -s $q $V/rollouts/; done
    python3 -c "import json,sys,glob; json.dump({'rollouts': [r for f in sorted(glob.glob(sys.argv[1] + '/c*/run/aggregate/results-summary.json')) for r in json.load(open(f))['rollouts']]}, open(sys.argv[2], 'w'))" $W/read/$t $V/aggregate/results-summary.json
  done
  RO=$W/read; ZM=experiments/zeros_mech/scripts/zm_read.py
  env OMP_NUM_THREADS=1 $P $ZM collect --runs YSB-L18-s0=$RO/YSB-L18-s0/merged YSB-L18-s1=$RO/YSB-L18-s1/merged YS-L18-s0=$RO/c15-YS-L18-s0/merged YS-L18-s1=$RO/YS-L18-s1/run \
      YNONE-L18-s0=$RO/YNONE-L18-s0/run YNONE-L18-s1=$RO/YNONE-L18-s1/run YS2G-L18-s0=$RO/YS2G-L18-s0/run YS2G-L18-s1=$RO/YS2G-L18-s1/run \
      YR10m10-F-s0=$H/YR10m10-F-s0 YR10m10-F-s1=$H/YR10m10-F-s1 || die "zm_read collect rc $?"
  CD=$(ls -d $DATA_DIR/runs/zeros_mech/collect/*/ | tail -1); echo "$CD" > $W/report/Y-SB/zm_collect_dir.txt
  $P $ZM report --collect $CD --rows $(ls -d $DATA_DIR/runs/zeros_mech/rows/*/ | tail -1) --out $W/report/Y-SB/zm --figs $W/report/Y-SB/zm/figs || echo "zm_read report failed (run it on the Mac)"
  [[ -f $W/report/Y-SB/zm/per_rollout_reads.csv.gz ]] && $P $E/scripts/ol_dirraster.py lines --reads $W/report/Y-SB/zm --collect $CD --out $W/report/Y-SB/lines.json ;;
submit-armb2)  # Amendment 14: arm B', oncoming hinge channel (ol_dirraster.py onc -> label gate) -> YSO25 / YSO50 (--on-lon ol_onc, lam 2.5 / 5), seeds 0 / 1 ->
               # held reads in 6 chunks each (ol_fast.sh read), all off card 0 -> report-armb2
  sub() { local name=$1 after=$2; shift 2; local id; id=$(CL submit --owner onpol-long --name "$name" --tries 1 --priority ${PRIO:-14} --no-check ${after:+--after $after} \
            --log-dir "$W/pool/$name" "$@" | awk '{print $NF}') && [[ -n $id ]] || die "submit $name"; echo "$id"; echo "$id $name (amendment 14)" >> $W/jobs.txt; }
  RB=$W/rows/L18YS-onc; G15="--gpus 1,2,3,4,5 --pin-strict"; X1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
  d=$(sub ol-armb2-onc "" --vram 0.5 --cpu 64 --ram 120 --timeout-h 1 -- $X1 $NAV $E/scripts/ol_dirraster.py onc --out $RB --workers 64) || exit 1
  OT="env PYTHONPATH=$PWD/$E/lib OL_ONC=$RB/onc.npz $TR $ON/scripts/on_train.py train --exp onpol_long --init YS-L18-s1 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank"
  rids=()
  for x in "SO25 2.5" "SO50 5"; do set -- $x; A=$1; LAM=$2
    for s in 0 1; do
      a=$(sub ol-train-Y$A-s$s "$d" --train --vram 26 --cpu 6 --ram 60 --timeout-h 2 $G15 -- $OT --agent-lam 10 --agent-on-only --shape --ho-w 3 --on on1_L18YS --on-k 13 \
            --on-split alpasim_nuplan/onpol-train --on-mode self --on-lon ol_onc --on-lon-lam $LAM --seed $s --tag Y$A-L18-s$s) || exit 1
      for i in $(seq 0 5); do rids+=($(sub ol-read-Y$A-L18-s$s-$i "$a" --vram 32 --cpu 8 --ram 40 --timeout-h 2 $G15 --env CL_HIST_KEY=ol-fast-read -- \
            env OL_RCH=6 bash $E/scripts/ol_fast.sh read Y$A-L18-s$s $i 6)) || exit 1; done
    done
  done
  sub ol-report-armb2 "$(IFS=,; echo "${rids[*]}")" --vram 0.5 --cpu 64 --ram 64 --timeout-h 1 -- bash $E/scripts/ol_chain.sh report-armb2 > /dev/null || exit 1
  echo "submitted: $(grep -c "amendment 14" $W/jobs.txt) jobs" ;;
report-armb2)  # Amendment 14: each weight against YS2G and YS-L18 (seed means); zm_read (unedited, via ol_zm.py: Run root under runs/onpol_long/zm); the lines
  H=$DATA_DIR/runs/alpasim_long/held/h18; c() { ls -d $W/read/$1/c*/run | paste -sd,; }; O=$W/report/Y-SO; mkdir -p $O
  for A in SO25 SO50; do
    R=("Y$A-L18-s0@18=$(c Y$A-L18-s0)" "Y$A-L18-s1@18=$(c Y$A-L18-s1)" "YS2G-L18-s0@18=$W/read/YS2G-L18-s0/run" "YS2G-L18-s1@18=$W/read/YS2G-L18-s1/run"
       "YS-L18-s0@18=$(c c15-YS-L18-s0)" "YS-L18-s1@18=$W/read/YS-L18-s1/run")
    $P $E/scripts/ol_report.py --family Y --arm $A --ctl S2G --base-tags YS-L18-s0,YS-L18-s1 --ref-mean --runs "${R[@]}" --out $W/report/Y-$A || die "ol_report $A rc $?"
  done
  for t in YSO25-L18-s0 YSO25-L18-s1 YSO50-L18-s0 YSO50-L18-s1; do V=$W/read/$t/merged; rm -rf $V; mkdir -p $V/aggregate $V/driver-logs $V/rollouts
    cp $W/read/$t/c0/run/generated-user-config-0.yaml $V/; cat $W/read/$t/c*/run/driver-logs/drive.jsonl > $V/driver-logs/drive.jsonl
    for q in $W/read/$t/c*/run/rollouts/*; do ln -s $q $V/rollouts/; done
    python3 -c "import json,sys,glob; json.dump({'rollouts': [r for f in sorted(glob.glob(sys.argv[1] + '/c*/run/aggregate/results-summary.json')) for r in json.load(open(f))['rollouts']]}, open(sys.argv[2], 'w'))" $W/read/$t $V/aggregate/results-summary.json
  done
  RO=$W/read; ZW=$E/scripts/ol_zm.py
  env OMP_NUM_THREADS=1 $P $ZW collect --runs YSO25-L18-s0=$RO/YSO25-L18-s0/merged YSO25-L18-s1=$RO/YSO25-L18-s1/merged YSO50-L18-s0=$RO/YSO50-L18-s0/merged \
      YSO50-L18-s1=$RO/YSO50-L18-s1/merged YS-L18-s0=$RO/c15-YS-L18-s0/merged YS-L18-s1=$RO/YS-L18-s1/run YNONE-L18-s0=$RO/YNONE-L18-s0/run YNONE-L18-s1=$RO/YNONE-L18-s1/run \
      YS2G-L18-s0=$RO/YS2G-L18-s0/run YS2G-L18-s1=$RO/YS2G-L18-s1/run YR10m10-F-s0=$H/YR10m10-F-s0 YR10m10-F-s1=$H/YR10m10-F-s1 || die "zm_read collect rc $?"
  CD=$(ls -d $W/zm/zeros_mech/collect/*/ | tail -1); echo "$CD" > $O/zm_collect_dir.txt
  $P $ZW report --collect $CD --rows $(ls -d $DATA_DIR/runs/zeros_mech/rows/*/ | tail -1) --out $O/zm --figs $O/zm/figs || echo "zm_read report failed"
  [[ -f $O/zm/per_rollout_reads.csv.gz ]] && $P $E/scripts/ol_dirraster.py lines --reads $O/zm --collect $CD --out $O/lines.json ;;
submit-armb3)  # Amendment 15: oncoming region by zeros_mech's lane rule (ol_zmonc.py build, gates) -> arm B' YSO25 / YSO50 (--on-lon ol_onc) and, behind its own
               # gate (road_dir hinge <= 0.16), arm B redone YSBz (--on-road road_dir.npz); seeds 0 / 1; held reads 6 chunks each; report-armb3
  sub() { local name=$1 after=$2; shift 2; local id; id=$(CL submit --owner onpol-long --name "$name" --tries 1 --priority ${PRIO:-14} --no-check ${after:+--after $after} \
            --log-dir "$W/pool/$name-a15" "$@" | awk '{print $NF}') && [[ -n $id ]] || die "submit $name"; echo "$id"; echo "$id $name (amendment 15)" >> $W/jobs.txt; }
  RB=$W/rows/L18YS-zmonc; G15="--gpus 1,2,3,4,5 --pin-strict"; X1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1"
  d=$(sub ol-armb3-onc "" --vram 0.5 --cpu 64 --ram 160 --timeout-h 2 -- $X1 $P $E/scripts/ol_zmonc.py build --out $RB --workers 64 --armb $W/rows/L18YS-dir) || exit 1
  b=$(sub ol-armb3-bgate "$d" --vram 0.5 --cpu 1 --ram 4 --timeout-h 1 -- python3 -c "import json,sys; s = json.load(open('$RB/onc_stats.json')); print(s['roaddir_active']); sys.exit(0 if s['gate_b'] else 2)") || exit 1
  OT="$TR $ON/scripts/on_train.py train --exp onpol_long --init YS-L18-s1 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank \
      --agent-lam 10 --agent-on-only --shape --ho-w 3 --on on1_L18YS --on-k 13 --on-split alpasim_nuplan/onpol-train --on-mode self"
  rids=(); bids=()
  for x in "SO25 $d --on-lon ol_onc --on-lon-lam 2.5" "SO50 $d --on-lon ol_onc --on-lon-lam 5" "SBz $b --on-road $RB/road_dir.npz"; do set -- $x; A=$1; AF=$2; shift 2
    [[ $A == SBz ]] && { bids=("${rids[@]}"); rids=(); }
    for s in 0 1; do
      a=$(sub ol-train-Y$A-s$s "$AF" --train --vram 26 --cpu 6 --ram 60 --timeout-h 2 $G15 -- env PYTHONPATH=$PWD/$E/lib OL_ONC=$RB/onc.npz $OT "$@" --seed $s --tag Y$A-L18-s$s) || exit 1
      for i in $(seq 0 5); do rids+=($(sub ol-read-Y$A-L18-s$s-$i "$a" --vram 32 --cpu 8 --ram 40 --timeout-h 2 $G15 --env CL_HIST_KEY=ol-fast-read -- \
            env OL_RCH=6 bash $E/scripts/ol_fast.sh read Y$A-L18-s$s $i 6)) || exit 1; done
    done
  done
  # B' reads -> report; B's reads (skipped when its gate fails) -> the same report again, then with all three arms
  q=$(sub ol-report-armb3 "$(IFS=,; echo "${bids[*]}")" --vram 0.5 --cpu 64 --ram 64 --timeout-h 1 -- bash $E/scripts/ol_chain.sh report-armb3) || exit 1
  sub ol-report-armb3-all "$(IFS=,; echo "${rids[*]},$q")" --vram 0.5 --cpu 64 --ram 64 --timeout-h 1 -- bash $E/scripts/ol_chain.sh report-armb3 > /dev/null || exit 1
  echo "submitted: $(grep -c "amendment 15" $W/jobs.txt) jobs" ;;
report-armb3)  # Amendment 15: each arm present (SO25, SO50, SBz) against YS2G and YS-L18 (seed means); zm_read (unedited, Run root under runs/onpol_long/zm); the lines
  H=$DATA_DIR/runs/alpasim_long/held/h18; c() { ls -d $W/read/$1/c*/run | paste -sd,; }; O=$W/report/Y-A15; mkdir -p $O; ZR=()
  for A in SO25 SO50 SBz; do
    [[ $(ls $W/read/Y$A-L18-s0/c*/run/aggregate/results-summary.json $W/read/Y$A-L18-s1/c*/run/aggregate/results-summary.json 2>/dev/null | wc -l) -eq 12 ]] || { echo "skip $A: reads missing"; continue; }
    R=("Y$A-L18-s0@18=$(c Y$A-L18-s0)" "Y$A-L18-s1@18=$(c Y$A-L18-s1)" "YS2G-L18-s0@18=$W/read/YS2G-L18-s0/run" "YS2G-L18-s1@18=$W/read/YS2G-L18-s1/run"
       "YS-L18-s0@18=$(c c15-YS-L18-s0)" "YS-L18-s1@18=$W/read/YS-L18-s1/run")
    $P $E/scripts/ol_report.py --family Y --arm $A --ctl S2G --base-tags YS-L18-s0,YS-L18-s1 --ref-mean --runs "${R[@]}" --out $W/report/Y-$A || die "ol_report $A rc $?"
    for s in 0 1; do t=Y$A-L18-s$s; V=$W/read/$t/merged; rm -rf $V; mkdir -p $V/aggregate $V/driver-logs $V/rollouts
      cp $W/read/$t/c0/run/generated-user-config-0.yaml $V/; cat $W/read/$t/c*/run/driver-logs/drive.jsonl > $V/driver-logs/drive.jsonl
      for q in $W/read/$t/c*/run/rollouts/*; do ln -s $q $V/rollouts/; done
      python3 -c "import json,sys,glob; json.dump({'rollouts': [r for f in sorted(glob.glob(sys.argv[1] + '/c*/run/aggregate/results-summary.json')) for r in json.load(open(f))['rollouts']]}, open(sys.argv[2], 'w'))" $W/read/$t $V/aggregate/results-summary.json
      ZR+=("$t=$V"); done
  done
  RO=$W/read; ZW=$E/scripts/ol_zm.py
  env OMP_NUM_THREADS=1 $P $ZW collect --runs "${ZR[@]}" YS-L18-s0=$RO/c15-YS-L18-s0/merged YS-L18-s1=$RO/YS-L18-s1/run YNONE-L18-s0=$RO/YNONE-L18-s0/run YNONE-L18-s1=$RO/YNONE-L18-s1/run \
      YS2G-L18-s0=$RO/YS2G-L18-s0/run YS2G-L18-s1=$RO/YS2G-L18-s1/run YR10m10-F-s0=$H/YR10m10-F-s0 YR10m10-F-s1=$H/YR10m10-F-s1 || die "zm_read collect rc $?"
  CD=$(ls -d $W/zm/zeros_mech/collect/*/ | tail -1); echo "$CD" > $O/zm_collect_dir.txt
  $P $ZW report --collect $CD --rows $(ls -d $DATA_DIR/runs/zeros_mech/rows/*/ | tail -1) --out $O/zm --figs $O/zm/figs || echo "zm_read report failed"
  [[ -f $O/zm/per_rollout_reads.csv.gz ]] && $P $E/scripts/ol_dirraster.py lines --reads $O/zm --collect $CD --out $O/lines.json ;;
submit-armb4)  # Amendment 17: arm B' with variant (a) (rear-axle point, t0 heading) on the zeros_mech-rule raster: YSO25 / YSO50 x seeds 0 / 1, reads, report
  sub() { local name=$1 after=$2; shift 2; local id; id=$(CL submit --owner onpol-long --name "$name" --tries 1 --priority ${PRIO:-14} --no-check ${after:+--after $after} \
            --log-dir "$W/pool/$name-a17" "$@" | awk '{print $NF}') && [[ -n $id ]] || die "submit $name"; echo "$id"; echo "$id $name (amendment 17)" >> $W/jobs.txt; }
  RB=$W/rows/L18YS-zmonc; G15="--gpus 1,2,3,4,5 --pin-strict"
  python3 -c "import json,sys; s = json.load(open('$RB/onclab.json')); v = s['table']['a rear axle']['served']; print('served (a)', v); sys.exit(v > 0.10)" || die "label gate (a) > 0.10"
  OT="$TR $ON/scripts/on_train.py train --exp onpol_long --init YS-L18-s1 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank \
      --agent-lam 10 --agent-on-only --shape --ho-w 3 --on on1_L18YS --on-k 13 --on-split alpasim_nuplan/onpol-train --on-mode self --on-lon ol_onc"
  rids=()
  for x in "SO25 2.5" "SO50 5"; do set -- $x
    for s in 0 1; do
      a=$(sub ol-train-Y$1-s$s "" --train --vram 26 --cpu 6 --ram 60 --timeout-h 2 $G15 -- env PYTHONPATH=$PWD/$E/lib OL_ONC=$RB/onc.npz OL_ONC_MODE=point $OT --on-lon-lam $2 --seed $s --tag Y$1-L18-s$s) || exit 1
      for i in $(seq 0 5); do rids+=($(sub ol-read-Y$1-L18-s$s-$i "$a" --vram 32 --cpu 8 --ram 40 --timeout-h 2 $G15 --env CL_HIST_KEY=ol-fast-read -- \
            env OL_RCH=6 bash $E/scripts/ol_fast.sh read Y$1-L18-s$s $i 6)) || exit 1; done
    done
  done
  sub ol-report-armb4 "$(IFS=,; echo "${rids[*]}")" --vram 0.5 --cpu 64 --ram 64 --timeout-h 1 -- bash $E/scripts/ol_chain.sh report-armb3 > /dev/null || exit 1
  echo "submitted: $(grep -c "amendment 17" $W/jobs.txt) jobs" ;;
*) die "unknown stage $STAGE" ;;
esac
