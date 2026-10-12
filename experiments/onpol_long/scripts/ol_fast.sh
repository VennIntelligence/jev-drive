#!/usr/bin/env bash
# ONPOL-LONG round chain, fast variant (results/pipeline_profile.md). Same job bodies and outputs as ol_chain.sh (which stays as it is),
# changes that leave every per-scene result unchanged (bit-identity smoke in pipeline_profile.md: a rollout's plans do not depend on which
# other scenes share its simulator stack or card):
#   OL_NCH      train rollouts in this many pool jobs (default 12: two stacks per card; one stack used 10-40 % of a card)
#   OL_RCH      each held read in this many pool jobs (default 6); the report joins the chunk dirs (long_read takes comma-separated dirs)
#   OL_STEPS, OL_TRAIN_ARGS  trainer steps and extra on_train.py flags for all four trainings (default: 10 000, none = submit-r2's recipe)
#   OL_WORKERS / OL_LWORKERS  on_rows.py geo / ol_labels.py build workers (default 48 each, as ol_chain.sh; geo itself takes ~30 s)
#   and the K / C label build (ol_labels.py build, 18 of ol_chain.sh's 19 geo minutes) runs beside the tokens instead of before them: the S-only
#   recipe never reads its files; its gate (ol_labels.py gate, unchanged) still runs on every pool and gates strip and the report
# Dependencies: rollouts -> geo -> {tok, labels} ; tok -> arm trainings ; {tok, labels} -> gate -> strip ; each training -> its RCH reads ;
# all reads + gate -> report. The controls' trainings and reads start at once.
# Box, repo root:
#   bash experiments/onpol_long/scripts/ol_fast.sh submit-round <init tag> <arm> [<roll tags>]   -> one Y-recipe round (S rows), all stages queued
#   bash experiments/onpol_long/scripts/ol_fast.sh submit-smoke <tag> <scene list> <out dir>   -> one 18-decision read (pipeline smoke)
#   bash experiments/onpol_long/scripts/ol_fast.sh submit-scale <tag> <scene list> <out dir>    -> 1, 2, 3, 4 stacks at once on one card
# Job bodies (run by the pool): roll <T> <i> <n>; read <TAG> <i> <n>; readlist <TAG> <list> <dir>; geo <T,..> <N> <n>; labels <N>; tok <N> <ctl>;
# gate <N>; report <arm> <init> <out>.
# State as ol_chain.sh: $DATA_DIR/runs/onpol_long/{roll/<T>/c<i>/run, read/<TAG>/c<i>/run, rows/<N>, report/<name>}; lists under lists/n<n>/.
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/onpol_long; ON=experiments/onpol_nuplan; SPL=$ON/results/split; W=$DATA_DIR/runs/onpol_long; CK=$DATA_DIR/runs/op_parity/runs
C=$DATA_DIR/datasets/alpasim_nuplan_long/after160; KEY=runtime.scene_provider.trajdata
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; NAV=$DATA_DIR/envs/navsim2/bin/python; TR=$DATA_DIR/envs/op-train/bin/python
NCH=${OL_NCH:-12}; RCH=${OL_RCH:-6}; WK=${OL_WORKERS:-48}; LWK=${OL_LWORKERS:-48}
CL() { if [[ -n ${OL_DRY:-} ]]; then echo "DRY cl $*" >&2; echo "dry-$RANDOM"; else .venv/bin/python -m jevdrive.cl "$@"; fi; }   # OL_DRY=1: print the submissions
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
LONG=("$KEY.cache_location=$C" "$KEY.dataset.extra_params.num_timesteps_after=160")
split() {  # split <list> <n> <dir>: n interleaved chunks of a scene list, written once and never changed
  local L=$1 n=$2 d=$3; mkdir -p "$d"
  for i in $(seq 0 $((n - 1))); do awk -v i=$i -v n=$n '(NR - 1) % n == i' "$L" > "$d/c$i.txt.tmp"
    if [[ -f $d/c$i.txt ]]; then cmp -s "$d/c$i.txt" "$d/c$i.txt.tmp" || die "$d/c$i.txt changed"; rm "$d/c$i.txt.tmp"; else mv "$d/c$i.txt.tmp" "$d/c$i.txt"; fi
  done; }
lists() { [[ -z $(comm -12 <(sort $SPL/train.txt) <(sort $SPL/held.txt)) ]] || die "train and held overlap"
  [[ $(wc -l < $SPL/held.txt) -eq 460 ]] || die "held.txt"; split $SPL/train.txt $NCH $W/lists/n$NCH; split $SPL/held.txt $RCH $W/lists/held_n$RCH; }
prune() {  # ol_chain.sh read1's pruning: drop rollout.asl of scored, unfailed rollouts (roll: of every unfailed rollout)
  python3 - "$1" "$2" <<'PY'
import glob, json, os, sys
D, keep_zero = sys.argv[1], sys.argv[2] == "read"; n = 0
for r in json.load(open(f"{D}/aggregate/results-summary.json"))["rollouts"]:
    if not r.get("failure_reason") and (not keep_zero or (r.get("score") or 0) > 0):
        for f in glob.glob(f"{D}/rollouts/{r['clipgt_id']}/*/rollout.asl"):
            os.remove(f); n += 1
print("pruned", n, "rollout.asl")
PY
}
sim() {  # sim <tag> <scene list> <run dir> [dump]: one 18-decision run, ol_chain.sh's driver path and overrides
  local T=$1 L=$2 D=$3 DUMP=${4:-0}
  [[ -f $CK/$T/ckpt-final.pt ]] || die "no checkpoint $T"; [[ -f $C/nuplan_test/scenes_list.dill ]] || die "no cache $C"
  [[ ! -e $D/aggregate/results-summary.json ]] || die "$D already ran"
  rm -rf "$D"; mkdir -p "$D"
  env SH30_TAG=$T SH30_DUMP=$DUMP bash experiments/alpasim/scripts/run.sh "$D" sh30 --scene-list "$L" "${BASE[@]}" "${LONG[@]}" || die "run.sh rc $?"
  [[ -f $D/aggregate/results-summary.json ]] || die "no results-summary"; }
dirs() {  # the run dirs of one read, comma-joined: the chunk dirs, else ol_chain.sh's single run dir, else alpasim_long's h18 read
  local t=$1 c; c=$(ls -d $W/read/$t/c*/run 2>/dev/null | paste -sd,)
  [[ -z $c || $(ls $W/read/$t/c*/run/aggregate/results-summary.json | wc -l) -eq $(ls -d $W/read/$t/c* | wc -l) ]] || die "$t: a read chunk did not finish"
  if [[ -n $c ]]; then echo "$c"; elif [[ -f $W/read/$t/run/aggregate/results-summary.json ]]; then echo "$W/read/$t/run"; else echo "$DATA_DIR/runs/alpasim_long/held/h18/$t"; fi; }
STAGE=${1:?stage}; shift
case $STAGE in
roll)      # one chunk of the train split with the frame dump; then ol_chain.sh roll1's simulator-state extraction and pruning
  T=$1; i=$2; n=$3; D=$W/roll/$T/c$i/run; L=$W/lists/n$n/c$i.txt
  [[ -z $(comm -12 <(sort "$L") <(sort $SPL/held.txt)) ]] || die "$L has a held scene"
  sim "$T" "$L" "$D" 1000000
  $P experiments/alpasim/scripts/c1_extract.py logs --run "$D" --out "$D/onpol_sim.pkl" --jobs 8 || die "c1_extract rc $?"
  prune "$D" roll
  echo "$(date '+%F %T') roll $T c$i/$n done: $(ls $D/driver-logs/dump | wc -l) dumps, $(du -sh $D | cut -f1)" ;;
read)      # one chunk of the held 460 at 18 decisions
  TAG=$1; i=$2; n=$3; D=$W/read/$TAG/c$i/run
  sim "$TAG" "$W/lists/held_n$n/c$i.txt" "$D"; prune "$D" read ;;
readlist)  # one checkpoint on any scene list at 18 decisions (the pipeline smoke; OL_DUMP=1: with roll's frame dump, no pruning)
  if [[ -n ${OL_DUMP:-} ]]; then sim "$1" "$2" "$3" 1000000; else sim "$1" "$2" "$3"; prune "$3" read; fi ;;
geo)       # ol_chain.sh geo over n chunks, first half: manifest, on_rows.py geo (unedited), WK workers
  N=$2; n=$3; RR=()
  for T in ${1//,/ }; do R=$W/roll/$T; RR+=("$R")
  python3 - "$R" "$T" "$n" <<'PY' || die "rollout chunks missing ($T)"
import json, os, sys
R, T, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
assert not os.path.exists(f"{R}/c{n}"), f"{R} has more than {n} chunks"
m = {f"{T}@c{i}": [f"{R}/c{i}/run"] for i in range(n) if os.path.exists(f"{R}/c{i}/run/onpol_sim.pkl")}
json.dump(m, open(f"{R}/manifest.json", "w"), indent=1)
print(len(m), "of", n, "chunks"); sys.exit(len(m) != n)
PY
  done
  $NAV $ON/scripts/on_rows.py geo --name "$N" --roll "${RR[@]}" --workers $WK || die "on_rows geo rc $?" ;;
labels)    # ol_chain.sh geo, second half: K and C labels (ol_labels.py build); CPU
  $NAV $E/scripts/ol_labels.py build --name "$1" --workers $LWK || die "ol_labels build rc $?" ;;
tok)       # ol_chain.sh tok, first half: tokens + teacher + gate R1 (on_rows.py tok); GPU
  $TR $ON/scripts/on_rows.py tok --name "$1" --control "$2" || die "on_rows tok rc $?" ;;
gate)      # ol_chain.sh tok, second half: the pool gates (ol_labels.py gate: R1 / R2 as amended, K0) after tok and labels
  $TR $E/scripts/ol_labels.py gate --name "$1" --out "$W/rows/$1" || die "pool gates failed ($W/rows/$1/gates.md): the round's report stops here" ;;
report)    # ol_report.py for the Y family: arm Y<ARM>-L18-s{0,1}, control Y<ARM>NONE-L18-s{0,1}, against the round's start and the base YR10m10-F
  A=$1; I=$2; O=$3; I2=${I%-s*}-s$((1 - ${I##*-s})); R=()
  for t in Y$A-L18-s0 Y$A-L18-s1 Y${A}NONE-L18-s0 Y${A}NONE-L18-s1 $I $I2 YR10m10-F-s0 YR10m10-F-s1; do R+=("$t@18=$(dirs $t)"); done
  $P $E/scripts/ol_report.py --family Y --arm $A --ctl ${A}NONE --base-tags $I,$I2 --runs "${R[@]}" --out $W/report/$O || die "ol_report (start) rc $?"
  $P $E/scripts/ol_report.py --family Y --arm $A --ctl ${A}NONE --base-tags YR10m10-F-s0,YR10m10-F-s1 --runs "${R[@]}" --out $W/report/$O-base || die "ol_report (base) rc $?" ;;
scale)     # stacks per card: phases k = 1..4, each k stacks at once on this job's card, stack j on scenes 48 j .. 48 j + 47 of the list
  TAG=$1; L=$2; O=$3
  for k in 1 2 3 4; do t0=$(date +%s); pids=()
    for j in $(seq 0 $((k - 1))); do sed -n "$((48 * j + 1)),$((48 * j + 48))p" "$L" > $O/k$k-s$j.txt
      ( sim "$TAG" $O/k$k-s$j.txt $O/k$k-s$j > $O/k$k-s$j.log 2>&1 ) & pids+=($!); done
    wait "${pids[@]}"; echo "$(date '+%F %T') scale k=$k: $(( $(date +%s) - t0 )) s for $((48 * k)) scenes"
  done ;;
submit-smoke)
  TAG=$1; L=$2; D=$3
  CL submit --owner onpol-long --name ol-speed-smoke --tries 1 --priority ${PRIO:-13} --no-check --vram 24 --cpu 6 --ram 30 --timeout-h 1 \
     --log-dir "$D.pool" --env OL_DUMP=${OL_DUMP:-} -- bash $E/scripts/ol_fast.sh readlist "$TAG" "$L" "$D" ;;
submit-scale)
  TAG=$1; L=$2; O=$3; mkdir -p "$O"
  CL submit --owner onpol-long --name ol-speed-scale --tries 1 --priority ${PRIO:-13} --no-check --vram 79 --cpu 24 --ram 120 --timeout-h 2 \
     --log-dir "$O/pool" -- bash $E/scripts/ol_fast.sh scale "$TAG" "$L" "$O" ;;
submit-round)   # one Y-recipe round from <init> (submit-r2's recipe): rollouts of <roll tags> (default: init) on train in NCH jobs -> geo -> tok
                # (ol_chain.sh tok, gates) -> strip; arm Y<arm>-L18-s{0,1} after tok, control Y<arm>NONE-L18-s{0,1} at once; every checkpoint read
                # on held in RCH jobs right after its own training; report after all reads
  I=$1; A=$2; RT=${3:-$1}; N=L18$A; lists
  [[ -f $CK/$I/ckpt-final.pt || -n ${OL_DRY:-} ]] || die "no checkpoint $I"; [[ ! -e $W/rows/$N ]] || die "rows $N exist"
  # the arm's path (rollouts -> rows -> arm trainings -> arm reads) one priority step above the controls and the side stages
  sub() { local name=$1 after=$2; shift 2; local id; id=$(CL submit --owner onpol-long --name "$name" --tries 1 --priority ${SUBP:-${PRIO:-14}} --no-check ${after:+--after $after} \
            --log-dir "$W/pool/$name" "$@" | awk '{print $NF}') && [[ -n $id ]] || die "submit $name"; echo "$id"; [[ -n ${OL_DRY:-} ]] || echo "$id $name (ol_fast $A)" >> $W/jobs.txt; }
  HI=$((${PRIO:-14} + 1)); FX="env OL_NCH=$NCH OL_RCH=$RCH OL_WORKERS=$WK OL_LWORKERS=$LWK bash $E/scripts/ol_fast.sh"
  rr=()
  for T in ${RT//,/ }; do [[ -n ${OL_DRY:-} ]] || mkdir -p $W/roll/$T
    for i in $(seq 0 $((NCH - 1))); do rr+=($(SUBP=$HI sub ol-roll-$T-$i "" --vram 32 --cpu 8 --ram 40 --timeout-h 2 --env CL_HIST_KEY=ol-fast-roll -- $FX roll $T $i $NCH)) || exit 1; done
  done
  g=$(SUBP=$HI sub ol-geo-$A "$(IFS=,; echo "${rr[*]}")" --vram 0.5 --cpu $WK --ram 64 --timeout-h 2 -- $FX geo $RT $N $NCH) || exit 1
  l=$(sub ol-labels-$A "$g" --vram 0.5 --cpu $LWK --ram 96 --timeout-h 2 -- $FX labels $N) || exit 1
  k=$(SUBP=$HI sub ol-tok-$A "$g" --vram 16 --cpu 16 --ram 64 --timeout-h 2 -- $FX tok $N smoke10eY) || exit 1
  q=$(sub ol-gate-$A "$k,$l" --vram 0.5 --cpu 8 --ram 32 --timeout-h 1 -- $FX gate $N) || exit 1
  sub ol-strip-$A "$q" --vram 0.5 --cpu 16 --ram 16 --timeout-h 2 -- bash $E/scripts/ol_chain.sh strip $RT $N > /dev/null || exit 1
  OT="$TR $ON/scripts/on_train.py train --exp onpol_long --init $I --steps ${OL_STEPS:-10000} --yr-mass 0.1 --label-bank ${OL_TRAIN_ARGS:-}"   # e.g. Amendment 6's "--warmup 1 --lr-scale 0.1"
  rids=()
  for s in 0 1; do
    a=$(SUBP=$HI sub ol-train-Y$A-s$s "$k" --train --vram 26 --cpu 6 --ram 60 --timeout-h 4 -- $OT --agent-lam 10 --agent-on-only --shape --ho-w 3 --on on1_$N --on-k 13 \
          --on-split alpasim_nuplan/onpol-train --on-mode self --seed $s --tag Y$A-L18-s$s) || exit 1
    c=$(sub ol-train-Y${A}NONE-s$s "" --train --vram 26 --cpu 6 --ram 60 --timeout-h 4 -- $OT --seed $s --tag Y${A}NONE-L18-s$s) || exit 1
    for x in "$a Y$A-L18-s$s $HI" "$c Y${A}NONE-L18-s$s ${PRIO:-14}"; do set -- $x
      for i in $(seq 0 $((RCH - 1))); do rids+=($(SUBP=$3 sub ol-read-$2-$i "$1" --vram 32 --cpu 8 --ram 40 --timeout-h 2 --env CL_HIST_KEY=ol-fast-read -- $FX read $2 $i $RCH)) || exit 1; done; done
  done
  sub ol-report-Y$A "$(IFS=,; echo "${rids[*]}"),$q" --vram 0.5 --cpu 16 --ram 32 --timeout-h 1 -- $FX report $A $I Y-$A > /dev/null || exit 1
  echo "submitted: $(grep -c "(ol_fast $A)" $W/jobs.txt) jobs" ;;
*) die "unknown stage $STAGE" ;;
esac
