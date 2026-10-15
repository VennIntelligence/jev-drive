#!/usr/bin/env bash
# APPROACH COURSE (plans/2026-10-15-approach-course-prereg.md): the policy's own approach states (entry - 40 m .. entry) pulled toward the
# human's entry state (on_approach.py: L1 entry pose, L2 1 - 2 s blend x 2, L3 speed not above the human's), on top of RSG-L18's exact
# recipe (route_mem/train-RSG-L18-s0 meta.json), pool on1_APP = on1_L18RB0 (RSG's S rows) + the approach rows (rf_apppool.py).
#   A1 = --app-lam W                    A2 = A1 + arm b's lane hinge (--lane-lam 1 --lane-dz 0.5 --lane-neg 0.2, no --rm-order)
# Every GPU job through the pool at PRIO (default 12); ids appended to $O/jobs.txt. Box, repo root.
#   rf_app.sh build                     CPU: rf_apppool label nup (48 cores) + label pai, then pack -> on1_APP, dev.npz, results/approach/rows.*
#   rf_app.sh ident                     bit identity with every flag off: 50 eager steps of RSG-L18-s0's command with this code and with the
#                                       code before the approach course (git worktree at $BASE_REV), on_train ident + both dev ADE logs
#   [AFTER=ids] rf_app.sh pilot <W> [extra flags]   APP1P<W>-L18-s0: 1 000 steps, seed 0, batch 64 (on-k 10), gentle schedule, init RSG-L18-s0,
#                                       pool on1_APP, --app-lam W (W = 0: the control pilot, same rows without the terms), then rf_v2gate.py and
#                                       rf_appgate.py vs RSG-L18-s0
#   rf_app.sh train A1|A2 <W> <seed>    APP1-L18-s<seed> / APP2-L18-s<seed>: 2 500 gentle steps on RSG-L18-s<seed> (batch 128, on-k 20), the
#                                       gates vs the base, then the held 460 / 18-decision read (rm_chain.sh read, 6 chunks); read ids ->
#                                       $O/reads.txt, "<tag> <base>" -> $O/bases.txt
#   rf_app.sh report [tags]             one pool job after every read: held.py, ol_confirm paired vs each tag's base, rf_follow turn
#                                       completion, rf_appheld entry state of the held turns -> $O/report/
#   rf_app.sh guard <tag> ..            public 1491 / 10-decision guard (rf_v2.sh guard's block) -> $O/guard/
set -uo pipefail; cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/route_follow/approach; M=$DATA_DIR/runs/route_mem/read; CK=$DATA_DIR/runs/op_parity/runs
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; ME=experiments/route_follow/scripts/rf_app.sh
PRIO=${PRIO:-12}; BASE_REV=${BASE_REV:-107e2002}; RES=experiments/route_follow/results/approach
# RSG-L18-s0's command (meta.json argv) without --init / --steps / --seed / --tag (= rf_v2.sh REC)
REC="--warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-split alpasim_nuplan/onpol-train --on-mode self --on on1_L18RB0"
LANE="--lane-lam 1 --lane-dz 0.5 --lane-neg 0.2"
pool() { echo "--on on1_APP --on-weight $(python3 -c "import json; print(json.load(open('$RES/rows.json'))['on_weight'])")"; }
mkdir -p $O/pool
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner approach --name "$name" --tries 1 --priority $PRIO --no-check \
     --log-dir $O/pool/$name "$@" | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
gates() {  # tag base after: rf_v2gate (G-real / G-ST / G-over / G-rely), then rf_appgate
  local g; g=$(sub app-v2gate-$1 --after $3 --vram 20 --cpu 4 --ram 48 --timeout-h 1 -- $TR experiments/route_follow/scripts/rf_v2gate.py --tag $1 --base $2) || exit 1
  sub app-gate-$1 --after $g --vram 12 --cpu 4 --ram 48 --timeout-h 1 -- $TR experiments/route_follow/scripts/rf_appgate.py --tag $1 --base $2 > /dev/null || exit 1; }
case $1 in
build)
  a=$(sub app-lab-nup --vram 0.5 --cpu 48 --ram 120 --timeout-h 2 -- $TR experiments/route_follow/scripts/rf_apppool.py label --src nup --workers 48) || exit 1
  b=$(sub app-lab-pai --vram 0.5 --cpu 2 --ram 40 --timeout-h 1 -- $TR experiments/route_follow/scripts/rf_apppool.py label --src pai) || exit 1
  sub app-pack --after $a,$b --vram 0.5 --cpu 4 --ram 80 --timeout-h 1 -- $TR experiments/route_follow/scripts/rf_apppool.py pack > /dev/null || exit 1
  tail -3 $O/jobs.txt ;;
ident)
  [[ -d /tmp/jev-appbase ]] || git worktree add --detach /tmp/jev-appbase $BASE_REV > /dev/null || exit 1
  for t in APPID-new-s0 APPID-old-s0; do [[ ! -e $CK/$t ]] || { echo "$CK/$t exists"; exit 1; }; done
  a=$(sub app-ident-new --train --vram 28 --cpu 6 --ram 70 --timeout-h 1 -- bash $ME body-train APPID-new-s0 RSG-L18-s0 0 50 128 13 --eval-every 50 --allow-init) || exit 1
  b=$(sub app-ident-old --train --vram 28 --cpu 6 --ram 70 --timeout-h 1 --cwd /tmp/jev-appbase -- $TR experiments/route_mem/scripts/rm_train.py train --tag APPID-old-s0 \
      --init RSG-L18-s0 --steps 50 --batch 128 --on-k 13 --seed 0 $REC --eval-every 50 --allow-init) || exit 1
  sub app-ident-cmp --after $a,$b --vram 0.5 --cpu 1 --ram 8 --timeout-h 1 -- bash -c "$TR experiments/onpol_nuplan/scripts/on_train.py ident --a APPID-new-s0 --b APPID-old-s0 --out $O/ident.json; rc=\$?; \
      grep -h 'dev @' $DATA_DIR/runs/route_mem/train-APPID-new-s0/*/log.txt $DATA_DIR/runs/route_mem/train-APPID-old-s0/*/log.txt | tee $O/ident_dev.txt; exit \$rc" ;;
pilot)
  W=$2; T=APP1P${W/./}-L18-s0; F="$(pool) --app-lam $W"
  t=$(sub app-pilot-$W ${AFTER:+--after $AFTER} --train --vram 24 --cpu 6 --ram 70 --timeout-h 1 -- bash $ME body-train $T RSG-L18-s0 0 1000 64 10 $F "${@:3}" --allow-init) || exit 1
  gates $T RSG-L18-s0 $t; tail -3 $O/jobs.txt ;;
gate)  # gate <tag> [base]: both gates again (after a gate fix)
  gates $2 ${3:-RSG-L18-s0} ""; tail -2 $O/jobs.txt ;;
train)
  A=$2; W=$3; s=$4; B=RSG-L18-s$s; T=${TAG_PREFIX:-APP${A#A}}-L18-s$s
  case $A in A1) F="$(pool) --app-lam $W" ;; A2) F="$(pool) --app-lam $W $LANE" ;; *) echo "arm A1 | A2" >&2; exit 2 ;; esac
  [[ -f $CK/$B/ckpt-final.pt ]] || { echo "no base $B"; exit 1; }
  t=$(sub app-train-$T --train --vram 28 --cpu 6 --ram 70 --timeout-h 2 -- bash $ME body-train $T $B $s 2500 128 20 $F "${@:5}" --allow-init) || exit 1
  gates $T $B $t
  for i in 0 1 2 3 4 5; do
    sub app-read-$T-$i --after $t --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
        -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6 >> $O/reads.txt || exit 1
  done
  echo "$T $B" >> $O/bases.txt; tail -9 $O/jobs.txt ;;
report)
  shift; sub app-report --after "$(paste -sd, $O/reads.txt)" --vram 0.5 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME body-report "$@" > /dev/null || exit 1; tail -1 $O/jobs.txt ;;
guard)  # = rf_v2.sh guard, into $O/guard
  shift; G=$DATA_DIR/runs/route_follow/guard; W=$DATA_DIR/runs/onpol_long; dep=()
  for T in "$@"; do [[ ! -e $G/read10/$T ]] || { echo "$G/read10/$T exists"; exit 1; }
    for i in $(seq 0 9); do
      dep+=($(sub app-g10-$T-$i --vram 27 --cpu 8 --ram 30 --timeout-h 2 --env CL_HIST_KEY=ol-fast-read -- bash experiments/onpol_long/scripts/ol_chain.sh readlist $T \
            $W/lists/all_n10/c$i.txt 10 $G/read10/$T/b$i)) || exit 1
    done
  done
  sub app-guard --after "$(IFS=,; echo "${dep[*]}")" --vram 0.5 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME body-guard "$@" > /dev/null || exit 1; tail -1 $O/jobs.txt ;;
body-guard)
  shift; G=$DATA_DIR/runs/route_follow/guard; A=$DATA_DIR/runs/alpasim; Q=(); MN=(); MP=(); PH=()
  Q+=("YR10m10-F-s0@all10=$(python3 -c "import json; a = json.load(open('$A/ot3/a/manifest.json')); b = json.load(open('$A/cf1/a/manifest.json')); print(','.join(a['YR10m10-F-s0'] + a['YR10m10-F-s0@new'] + b['YR10m10-F-s0@fresh']))")")
  for T in "$@" RSG-L18-s0 RSG-L18-s1; do Q+=("$T@all10=$(ls -d $G/read10/$T/b*/ | sed 's#/$##' | paste -sd,)"); done
  for f in $(for T in "$@"; do echo ${T%-s*}; done | awk '!s[$0]++'); do MN+=("$f=$f-s0,$f-s1"); MP+=("$f:RSG-L18"); done
  MN+=("RSG-L18=RSG-L18-s0,RSG-L18-s1"); for T in "$@"; do PH+=("$T:RSG-L18-s${T##*-s}"); done
  $P experiments/onpol_long/scripts/ol_confirm.py --runs "${Q[@]}" --base YR10m10-F-s0 --mean "${MN[@]}" --pairs "${MP[@]}" "${PH[@]}" --out $O/guard ;;
body-train)
  T=$2; I=$3; s=$4; N=$5; BS=$6; K=$7; shift 7
  exec $TR experiments/route_mem/scripts/rm_train.py train --tag $T --init $I --steps $N --batch $BS --on-k $K --seed $s $REC "$@" ;;
body-report)
  shift; TAGS="${*:-$(cut -d' ' -f1 $O/bases.txt | awk '!s[$0]++' | paste -sd' ')}"
  REF=$(for t in $TAGS; do grep "^$t " $O/bases.txt | tail -1 | cut -d' ' -f2; done | awk 'NF && !s[$0]++' | paste -sd' ')
  ALL=$(for t in $TAGS $REF; do ls $M/$t/c*/run/aggregate/results-summary.json > /dev/null 2>&1 && echo $t; done | awk '!s[$0]++' | paste -sd' ')
  mkdir -p $O/report; R=""; H=(); F=(); PH=(); PF=(); MN=(); MP=()
  for t in $ALL; do d=$(ls -d $M/$t/c*/run | paste -sd,); R="$R $t=$d"; H+=("$t@held18=$d"); F+=("$t@18=$d"); done
  for t in $TAGS; do b=$(grep "^$t " $O/bases.txt | tail -1 | cut -d' ' -f2)
    [[ " $ALL " == *" $b "* && $b != $t ]] && { PH+=("$t:$b"); PF+=("$t@18:$b@18"); }; done
  for f in $(for t in $TAGS; do echo ${t%-s*}; done | awk '!s[$0]++'); do
    [[ " $ALL " == *" $f-s0 "* && " $ALL " == *" $f-s1 "* ]] && { MN+=("$f=$f-s0,$f-s1"); MP+=("$f:RSG-L18"); }; done
  [[ ${#MN[@]} -gt 0 ]] && MN+=("RSG-L18=RSG-L18-s0,RSG-L18-s1")
  python3 $DATA_DIR/runs/route_follow/held.py $R > $O/report/held_raw.txt; rc0=$?
  $P experiments/onpol_long/scripts/ol_confirm.py --runs "${H[@]}" --base RSG-L18-s0 ${MN:+--mean "${MN[@]}"} --pairs "${MP[@]}" "${PH[@]}" --out $O/report/held18 > /dev/null; rc1=$?
  $P experiments/route_follow/scripts/rf_follow.py --runs "${F[@]}" --out $O/report/follow --workers 16 --pairs "${PF[@]}" > /dev/null; rc2=$?
  $P experiments/route_follow/scripts/rf_appheld.py --follow $O/report/follow --out $O/report/entry --pairs "${PF[@]}"; rc3=$?
  echo "held $rc0 held18 $rc1 follow $rc2 entry $rc3"; exit $((rc0 + rc1 + rc2 + rc3)) ;;
*) echo "rf_app.sh build | ident | pilot <W> | gate <tag> | train A1|A2 <W> <seed> | report [tags] | guard <tag> .." >&2; exit 2 ;;
esac
