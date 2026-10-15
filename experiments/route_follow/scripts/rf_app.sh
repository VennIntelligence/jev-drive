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
# APPROACH COURSE v2 (plans/2026-10-15-approach-v2-prereg.md; same pool on1_APP, + sidecar approach_v2.npz, rm_train --app-v2), into $O/v2:
#   rf_app.sh v2rows                    CPU: keep the v1 labels as lab_<src>.v1.npz, relabel both sources (adds Hvh, the v1 keys must be
#                                       equal), rf_apppool.py v2 -> on1_APP/approach_v2.npz, results/approach/v2/rows.*
#   [BASE_REV=rev] rf_app.sh ident <suffix> [flags]   bit identity of this code vs the worktree at BASE_REV (/tmp/jev-appbase-<rev>), 50 eager
#                                       steps of RSG-L18-s0's command + flags -> $O/v2/ident<suffix>.json
#   [AFTER=ids] rf_app.sh pilot2 <W>    APP2P<W>-L18-s0: the v1 pilot recipe with --app-lam W --app-v2; guard gates only (G-real <= base + 0.015)
#   rf_app.sh train2 <W> <seed>         APP2W<W>-L18-s<seed>: 2 500 gentle steps + guard gates + held 460 read (6 chunks) -> $O/v2/{reads,bases}.txt
#   rf_app.sh report2 [tags]            the report stage into $O/v2/report
#   rf_app.sh control <seed>            v2 Amendment 3, the W 0 control: APP0C-L18-s<seed> = 2 500 gentle steps on pool on1_APP, --app-lam 0
#                                       (approach rows with the S hinges only), reported guard gates, held 460 read -> $O/w0/{reads,bases}.txt
#                                       (bases: RSG-L18-s<seed> and APP2W05-L18-s0)
#   rf_app.sh report0 [tags]            the report stage into $O/w0/report (every base line of a tag is paired)
set -uo pipefail; cd "$(dirname "$0")/../../.."
O=${APP_O:-$DATA_DIR/runs/route_follow/approach}; O2=$DATA_DIR/runs/route_follow/approach/v2; O3=$DATA_DIR/runs/route_follow/approach/w0; M=$DATA_DIR/runs/route_mem/read; CK=$DATA_DIR/runs/op_parity/runs
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; ME=experiments/route_follow/scripts/rf_app.sh
PRIO=${PRIO:-12}; BASE_REV=${BASE_REV:-107e2002}; RES=experiments/route_follow/results/approach
# RSG-L18-s0's command (meta.json argv) without --init / --steps / --seed / --tag (= rf_v2.sh REC)
REC="--warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-split alpasim_nuplan/onpol-train --on-mode self --on on1_L18RB0"
LANE="--lane-lam 1 --lane-dz 0.5 --lane-neg 0.2"
pool() { echo "--on on1_APP --on-weight $(python3 -c "import json; print(json.load(open('$RES/rows.json'))['on_weight'])")"; }
mkdir -p $O/pool $O2 $O3
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner approach --name "$name" --tries 1 --priority $PRIO --no-check \
     --log-dir $O/pool/$name "$@" | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
gates() {  # tag base after: rf_v2gate (G-real / G-ST / G-over / G-rely), then rf_appgate
  local g; g=$(sub app-v2gate-$1 --after $3 --vram 20 --cpu 4 --ram 48 --timeout-h 1 -- $TR experiments/route_follow/scripts/rf_v2gate.py --tag $1 --base $2) || exit 1
  sub app-gate-$1 --after $g --vram 12 --cpu 4 --ram 48 --timeout-h 1 -- $TR experiments/route_follow/scripts/rf_appgate.py --tag $1 --base $2 ${GATE_FLAGS:-} > /dev/null || exit 1; }
G2="--guards-only --real-tol 0.015 --res-sub v2/gate"
case $1 in
build)
  a=$(sub app-lab-nup --vram 0.5 --cpu 48 --ram 120 --timeout-h 2 -- $TR experiments/route_follow/scripts/rf_apppool.py label --src nup --workers 48) || exit 1
  b=$(sub app-lab-pai --vram 0.5 --cpu 2 --ram 40 --timeout-h 1 -- $TR experiments/route_follow/scripts/rf_apppool.py label --src pai) || exit 1
  sub app-pack --after $a,$b --vram 0.5 --cpu 4 --ram 80 --timeout-h 1 -- $TR experiments/route_follow/scripts/rf_apppool.py pack > /dev/null || exit 1
  tail -3 $O/jobs.txt ;;
ident)  # ident <suffix> [flags]: this code vs BASE_REV, both with RSG-L18-s0's command + flags (none: every flag off)
  X=${2:?suffix}; shift 2; WT=/tmp/jev-appbase-$BASE_REV; NEW=APPID$X-new-s0; OLD=APPID$X-old-s0
  [[ -d $WT ]] || git worktree add --detach $WT $BASE_REV > /dev/null || exit 1
  for t in $NEW $OLD; do [[ ! -e $CK/$t ]] || { echo "$CK/$t exists"; exit 1; }; done
  a=$(sub app-ident-$X-new --train --vram 28 --cpu 6 --ram 70 --timeout-h 1 -- bash $ME body-train $NEW RSG-L18-s0 0 50 128 13 --eval-every 50 --allow-init "$@") || exit 1
  b=$(sub app-ident-$X-old --train --vram 28 --cpu 6 --ram 70 --timeout-h 1 --cwd $WT -- $TR experiments/route_mem/scripts/rm_train.py train --tag $OLD \
      --init RSG-L18-s0 --steps 50 --batch 128 --on-k 13 --seed 0 $REC --eval-every 50 --allow-init "$@") || exit 1
  sub app-ident-$X-cmp --after $a,$b --vram 0.5 --cpu 1 --ram 8 --timeout-h 1 -- bash -c "$TR experiments/onpol_nuplan/scripts/on_train.py ident --a $NEW --b $OLD --out $O2/ident$X.json; rc=\$?; \
      grep -h 'dev @' $DATA_DIR/runs/route_mem/train-$NEW/*/log.txt $DATA_DIR/runs/route_mem/train-$OLD/*/log.txt | tee $O2/ident${X}_dev.txt; exit \$rc" ;;
v2rows)
  for s in nup pai; do [[ -f $O/lab_$s.v1.npz ]] || cp $O/lab_$s.npz $O/lab_$s.v1.npz || exit 1; done
  a=$(sub app2-lab-nup --vram 0.5 --cpu 48 --ram 120 --timeout-h 2 -- $TR experiments/route_follow/scripts/rf_apppool.py label --src nup --workers 48) || exit 1
  b=$(sub app2-lab-pai --vram 0.5 --cpu 2 --ram 40 --timeout-h 1 -- $TR experiments/route_follow/scripts/rf_apppool.py label --src pai) || exit 1
  sub app2-v2 --after $a,$b --vram 0.5 --cpu 4 --ram 40 --timeout-h 1 -- $TR experiments/route_follow/scripts/rf_apppool.py v2 > /dev/null || exit 1
  tail -3 $O/jobs.txt ;;
pilot2)
  W=$2; T=APP2P${W/./}-L18-s0; F="$(pool) --app-lam $W --app-v2"
  t=$(sub app2-pilot-$W ${AFTER:+--after $AFTER} --train --vram 24 --cpu 6 --ram 70 --timeout-h 1 -- bash $ME body-train $T RSG-L18-s0 0 1000 64 10 $F "${@:3}" --allow-init) || exit 1
  GATE_FLAGS=$G2 gates $T RSG-L18-s0 $t; tail -3 $O/jobs.txt ;;
gate2)  # gate2 <tag> [base]: the v2 guard gates again
  GATE_FLAGS=$G2 gates $2 ${3:-RSG-L18-s0} ""; tail -2 $O/jobs.txt ;;
train2)
  W=$2; s=$3; B=RSG-L18-s$s; T=APP2W${W/./}-L18-s$s; F="$(pool) --app-lam $W --app-v2"
  [[ -f $CK/$B/ckpt-final.pt ]] || { echo "no base $B"; exit 1; }
  t=$(sub app2-train-$T ${AFTER:+--after $AFTER} --train --vram 28 --cpu 6 --ram 70 --timeout-h 2 -- bash $ME body-train $T $B $s 2500 128 20 $F "${@:4}" --allow-init) || exit 1
  GATE_FLAGS=$G2 gates $T $B $t
  for i in 0 1 2 3 4 5; do
    sub app2-read-$T-$i --after $t --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
        -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6 >> $O2/reads.txt || exit 1
  done
  echo "$T $B" >> $O2/bases.txt; tail -9 $O/jobs.txt ;;
control)
  s=$2; B=RSG-L18-s$s; T=APP0C-L18-s$s; F="$(pool) --app-lam 0"
  [[ ! -e $CK/$T ]] || { echo "$CK/$T exists"; exit 1; }
  t=$(sub app0-train-$T --train --vram 28 --cpu 6 --ram 70 --timeout-h 2 -- bash $ME body-train $T $B $s 2500 128 20 $F --allow-init) || exit 1
  GATE_FLAGS="--guards-only --real-tol 0.015 --res-sub w0/gate" gates $T $B $t
  for i in 0 1 2 3 4 5; do
    sub app0-read-$T-$i --after $t --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
        -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6 >> $O3/reads.txt || exit 1
  done
  printf '%s %s\n%s %s\n' $T $B $T APP2W05-L18-s0 >> $O3/bases.txt; tail -9 $O/jobs.txt ;;
report0)
  shift; sub app0-report --after "$(paste -sd, $O3/reads.txt)" --env APP_O=$O3 --vram 0.5 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME body-report "$@" > /dev/null || exit 1
  tail -1 $O/jobs.txt ;;
report2)
  shift; sub app2-report --after "$(paste -sd, $O2/reads.txt)" --env APP_O=$O2 --vram 0.5 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME body-report "$@" > /dev/null || exit 1
  tail -1 $O/jobs.txt ;;
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
  REF=$(for t in $TAGS; do grep "^$t " $O/bases.txt | cut -d' ' -f2; done | awk 'NF && !s[$0]++' | paste -sd' ')
  ALL=$(for t in $TAGS $REF; do ls $M/$t/c*/run/aggregate/results-summary.json > /dev/null 2>&1 && echo $t; done | awk '!s[$0]++' | paste -sd' ')
  mkdir -p $O/report; R=""; H=(); F=(); PH=(); PF=(); MN=(); MP=()
  for t in $ALL; do d=$(ls -d $M/$t/c*/run | paste -sd,); R="$R $t=$d"; H+=("$t@held18=$d"); F+=("$t@18=$d"); done
  for t in $TAGS; do for b in $(grep "^$t " $O/bases.txt | cut -d' ' -f2 | awk '!s[$0]++'); do       # every base line of the tag
    [[ " $ALL " == *" $b "* && $b != $t ]] && { PH+=("$t:$b"); PF+=("$t@18:$b@18"); }; done; done
  for f in $(for t in $TAGS; do echo ${t%-s*}; done | awk '!s[$0]++'); do
    [[ " $ALL " == *" $f-s0 "* && " $ALL " == *" $f-s1 "* ]] && { MN+=("$f=$f-s0,$f-s1"); MP+=("$f:RSG-L18"); }; done
  [[ ${#MN[@]} -gt 0 ]] && MN+=("RSG-L18=RSG-L18-s0,RSG-L18-s1")
  python3 $DATA_DIR/runs/route_follow/held.py $R > $O/report/held_raw.txt; rc0=$?
  $P experiments/onpol_long/scripts/ol_confirm.py --runs "${H[@]}" --base RSG-L18-s0 ${MN:+--mean "${MN[@]}"} --pairs "${MP[@]}" "${PH[@]}" --out $O/report/held18 > /dev/null; rc1=$?
  $P experiments/route_follow/scripts/rf_follow.py --runs "${F[@]}" --out $O/report/follow --workers 16 --pairs "${PF[@]}" > /dev/null; rc2=$?
  $P experiments/route_follow/scripts/rf_appheld.py --follow $O/report/follow --out $O/report/entry --pairs "${PF[@]}"; rc3=$?
  echo "held $rc0 held18 $rc1 follow $rc2 entry $rc3"; exit $((rc0 + rc1 + rc2 + rc3)) ;;
*) echo "rf_app.sh build | ident <suffix> [flags] | pilot <W> | gate <tag> | train A1|A2 <W> <seed> | report [tags] | guard <tag> .. | v2rows | pilot2 <W> | gate2 <tag> | train2 <W> <seed> | report2 [tags] | control <seed> | report0 [tags]" >&2; exit 2 ;;
esac
