#!/usr/bin/env bash
# ROUTE v2 (plans/2026-10-15-route-v2-prereg.md, Amendment 2): arms on top of RSG-L18-s0's exact recipe (route_mem/train-RSG-L18-s0 meta.json)
#   V2a = --rm-order --lc-cf --lc-lam 1   (ordered, progress-anchored route memory: the slot defect fix of decision 299; lane-change pairs)
#   V2b = V2a + --lane-lam 1              (near-field lane hinge with shift / straight negatives and the over-turn guard)
# Every GPU job through the pool at low priority (PRIO, default 6); ids appended to $O/jobs.txt. Box, repo root.
#   rf_v2.sh build                      CPU: rf_cfpool build --order (12 shards) -> rm_build order (12 navtrain shards + yaw-rate rows, pool
#                                       on1_L18RB0, on1_CF / on1_CFdev) ; G-slot memory of the held cases (rf_v2gate slotmem)
#   rf_v2.sh ident                      bit identity with every flag off: 50 eager steps of RSG-L18-s0's command with this code and with the
#                                       code before route v2 (git worktree at $BASE_REV), then on_train ident + the dev ADE of both logs
#   rf_v2.sh cpool                      CPU: arm c's pool on1_L18RB0J (rf_v2cpool.py)
#   [AFTER=ids] [PSFX=s] rf_v2.sh pilot a|b|c|n|o|l [extra flags]   RV2<arm>P<s>-L18-s0 (n = no flag, o = --rm-order, l = --lc-cf: diagnostics;
#                                       the control pilot of Amendment 3 ran as arm "c" before arm c existed: RV2cP-L18-s0 = no flag): 1 000 steps, seed 0, batch 64 (on-k 7, lc-k 4), gentle schedule, init
#                                       RSG-L18-s0, the arm's flags, then rf_v2gate.py vs RSG-L18-s0
#   rf_v2.sh train a|b|c <base> <seed> [extra flags]    RV2a-L18-s<seed> / RV2b-L18-s<seed> (TAG_PREFIX overrides RV2a / RV2b): 2 500 gentle steps on <base>
#                                       (--allow-init), the gate vs <base>, then the held 460 / 18-decision read (rm_chain.sh read, 6 chunks);
#                                       read ids -> $O/reads.txt, "<tag> <base>" -> $O/bases.txt; rm_build served on chunk 0 (G-serve:
#                                       the served ordered memory equals the offline rebuild -> runs/route_mem/served/<tag>.json)
#   rf_v2.sh report [tags]              one pool job after every read in $O/reads.txt: held.py, ol_confirm.py paired vs each tag's base and
#                                       RSG-L18-s<seed>, rf_follow.py turn completion -> $O/report/
#   rf_v2.sh guard <tag> ..             public 1491 / 10-decision guard (H3), rf_round2.sh's block: 10 read10 chunks per tag + ol_confirm vs
#                                       YR10m10-F-s0 with 2-seed means and pairs vs RSG-L18(-s<seed>) -> $O/guard/
#   rf_v2.sh body-train <tag> <init> <seed> <steps> <batch> <on-k> [flags] | body-report [tags]   (job bodies)
set -uo pipefail; cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/route_follow/v2; M=$DATA_DIR/runs/route_mem/read; CK=$DATA_DIR/runs/op_parity/runs
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; ME=experiments/route_follow/scripts/rf_v2.sh
PRIO=${PRIO:-6}; BASE_REV=${BASE_REV:-6551f10b}
# RSG-L18-s0's command (meta.json argv) without --init / --steps / --seed / --tag
REC="--warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-split alpasim_nuplan/onpol-train --on-mode self --on on1_L18RB0"
ARM_a="--rm-order"; ARM_b="--rm-order --lane-lam 1 --lane-dz 0.5 --lane-neg 0.2"   # Amendment 6: lc-cf dropped (fails G-LC at every guard-passing weight)
# arm c (Amendment 7): --rm-order + the junction counterfactual rows of on1_CF (rf_v2cpool.py: pool on1_L18RB0J = RSG's S rows + j0 / j1),
# RB-F's recipe on them (on_route hinge, dz 0.5, lateral only + road hinge, no imitation); ~12 of 25 pool draws are CF rows (pool=cf weight)
ARM_c="--rm-order --on on1_L18RB0J --on-weight pool=cf:1.3213 --on-lon on_route"
arm() { case $1 in a) echo "$ARM_a" ;; b) echo "$ARM_b" ;; c) echo "$ARM_c" ;; n) echo "" ;; o) echo "--rm-order" ;; l) echo "--lc-cf" ;; *) echo "arm a | b | c (| n o l: diagnostics)" >&2; exit 2 ;; esac; }
onk() { [[ $1 == c ]] && echo $(( $2 * 25 / 13 )) || echo $2; }   # arm c: 13 S + 12 CF rows per 128
D12=$(for i in $(seq 0 11); do printf "navtrain_full.s%dof12 " $i; done)
mkdir -p $O/pool
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner route-follow --name "$name" --tries 1 --priority $PRIO --no-check \
     --log-dir $O/pool/$name "$@" | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
case $1 in
cpool)
  sub v2-cpool --vram 0.5 --cpu 4 --ram 60 --timeout-h 1 -- $TR experiments/route_follow/scripts/rf_v2cpool.py --out L18RB0J ;;
build)
  c=$(sub v2-cforder --vram 0.5 --cpu 24 --ram 60 --timeout-h 1 -- $P experiments/route_follow/scripts/rf_cfpool.py build --order --data $D12 --workers 24 --force) || exit 1
  sub v2-order --after $c --vram 0.5 --cpu 24 --ram 60 --timeout-h 1 -- $P experiments/route_mem/scripts/rm_build.py order --data $D12 --pool L18RB0 --cf on1_CF on1_CFdev --workers 24 || exit 1
  for s in 0 1; do
    sub v2-slotmem-s$s --vram 0.5 --cpu 2 --ram 16 --timeout-h 1 -- $P experiments/route_follow/scripts/rf_v2gate.py slotmem --set held_s$s \
        --runs $(ls -d $M/RSG-L18-s$s/c*/run) --out $O/slotmem_held_s$s.json || exit 1
  done ;;
ident)
  [[ -d /tmp/jev-v2base ]] || git worktree add --detach /tmp/jev-v2base $BASE_REV > /dev/null || exit 1
  for t in RV2ID-new-s0 RV2ID-old-s0; do [[ ! -e $CK/$t ]] || { echo "$CK/$t exists"; exit 1; }; done
  a=$(sub v2-ident-new --train --vram 28 --cpu 6 --ram 70 --timeout-h 1 -- bash $ME body-train RV2ID-new-s0 RSG-L18-s0 0 50 128 13 --eval-every 50 --allow-init) || exit 1
  b=$(sub v2-ident-old --train --vram 28 --cpu 6 --ram 70 --timeout-h 1 --cwd /tmp/jev-v2base -- $TR experiments/route_mem/scripts/rm_train.py train --tag RV2ID-old-s0 \
      --init RSG-L18-s0 --steps 50 --batch 128 --on-k 13 --seed 0 $REC --eval-every 50 --allow-init) || exit 1
  sub v2-ident-cmp --after $a,$b --vram 0.5 --cpu 1 --ram 8 --timeout-h 1 -- bash -c "$TR experiments/onpol_nuplan/scripts/on_train.py ident --a RV2ID-new-s0 --b RV2ID-old-s0 --out $O/ident.json; rc=\$?; \
      grep -h 'dev @' $DATA_DIR/runs/route_mem/train-RV2ID-new-s0/*/log.txt $DATA_DIR/runs/route_mem/train-RV2ID-old-s0/*/log.txt | tee $O/ident_dev.txt; exit \$rc" ;;
pilot)
  A=$2; F=$(arm $A) || exit 1; T=RV2${A}P${PSFX:-}-L18-s0
  t=$(sub v2-pilot-$A${PSFX:-} ${AFTER:+--after $AFTER} --train --vram 24 --cpu 6 --ram 70 --timeout-h 1 -- bash $ME body-train $T RSG-L18-s0 0 1000 64 $(onk $A 7) $F --lc-k 4 "${@:3}" --allow-init) || exit 1
  sub v2-gate-$T --after $t --vram 20 --cpu 4 --ram 48 --timeout-h 1 -- $TR experiments/route_follow/scripts/rf_v2gate.py --tag $T --base RSG-L18-s0 > /dev/null || exit 1
  tail -2 $O/jobs.txt ;;
gate)  # gate <tag> [base]: the gate alone (re-read after a gate fix)
  sub v2-gate-$2-r$(date +%H%M) --vram 20 --cpu 4 --ram 48 --timeout-h 1 -- $TR experiments/route_follow/scripts/rf_v2gate.py --tag $2 --base ${3:-RSG-L18-s0} > /dev/null || exit 1
  tail -1 $O/jobs.txt ;;
train)
  A=$2; B=$3; s=$4; F=$(arm $A) || exit 1; T=${TAG_PREFIX:-RV2$A}-L18-s$s
  [[ -f $CK/$B/ckpt-final.pt ]] || { echo "no base $B"; exit 1; }
  t=$(sub v2-train-$T --train --vram 28 --cpu 6 --ram 70 --timeout-h 2 -- bash $ME body-train $T $B $s 2500 128 $(onk $A 13) $F --lc-k 8 "${@:5}" --allow-init) || exit 1
  sub v2-gate-$T --after $t --vram 20 --cpu 4 --ram 48 --timeout-h 1 -- $TR experiments/route_follow/scripts/rf_v2gate.py --tag $T --base $B > /dev/null || exit 1
  for i in 0 1 2 3 4 5; do
    sub v2-read-$T-$i --after $t --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
        -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6 >> $O/reads.txt || exit 1
  done
  r0=$(tail -6 $O/reads.txt | head -1)   # G-serve of the ordered memory: served (drive.jsonl rmem) vs the offline rebuild, chunk 0
  sub v2-served-$T --after $r0 --vram 0.5 --cpu 8 --ram 16 --timeout-h 1 -- $P experiments/route_mem/scripts/rm_build.py served --run $M/$T/c0/run --tag $T --workers 8 > /dev/null || exit 1
  echo "$T $B" >> $O/bases.txt; tail -8 $O/jobs.txt ;;
report)
  shift; sub v2-report --after "$(paste -sd, $O/reads.txt)" --vram 0.5 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME body-report "$@" > /dev/null || exit 1; tail -1 $O/jobs.txt ;;
guard)  # guard <tag> ..: public 1491 / 10 decisions (rf_round2.sh's guard block): 10 readlist chunks per tag, then one guard report job
  shift; G=$DATA_DIR/runs/route_follow/guard; W=$DATA_DIR/runs/onpol_long; dep=()
  for T in "$@"; do [[ ! -e $G/read10/$T ]] || { echo "$G/read10/$T exists"; exit 1; }
    for i in $(seq 0 9); do
      dep+=($(sub v2-g10-$T-$i --vram 27 --cpu 8 --ram 30 --timeout-h 2 --env CL_HIST_KEY=ol-fast-read -- bash experiments/onpol_long/scripts/ol_chain.sh readlist $T \
            $W/lists/all_n10/c$i.txt 10 $G/read10/$T/b$i)) || exit 1
    done
  done
  sub v2-guard --after "$(IFS=,; echo "${dep[*]}")" --vram 0.5 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME body-guard "$@" > /dev/null || exit 1; tail -1 $O/jobs.txt ;;
body-guard)  # ol_confirm vs YR10m10-F-s0 (the stored 2026-10-09 rollouts), 2-seed means, pairs vs RSG-L18-s<seed> -> $O/guard/
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
  REF=$(for t in $TAGS; do grep "^$t " $O/bases.txt | tail -1 | cut -d' ' -f2; echo "RSG-L18-s${t##*-s}"; done | awk 'NF && !s[$0]++' | paste -sd' ')
  ALL=$(for t in $TAGS $REF; do ls $M/$t/c*/run/aggregate/results-summary.json > /dev/null 2>&1 && echo $t; done | awk '!s[$0]++' | paste -sd' ')
  mkdir -p $O/report; R=""; H=(); F=(); PH=(); PF=()
  for t in $ALL; do d=$(ls -d $M/$t/c*/run | paste -sd,); R="$R $t=$d"; H+=("$t@held18=$d"); F+=("$t@18=$d"); done
  for t in $TAGS; do b=$(grep "^$t " $O/bases.txt | tail -1 | cut -d' ' -f2)
    for r in $b RSG-L18-s${t##*-s}; do [[ " $ALL " == *" $r "* && $r != $t ]] && { PH+=("$t:$r"); PF+=("$t@18:$r@18"); }; done; done
  python3 $DATA_DIR/runs/route_follow/held.py $R > $O/report/held_raw.txt; rc0=$?
  $P experiments/onpol_long/scripts/ol_confirm.py --runs "${H[@]}" --base RSG-L18-s0 --pairs "${PH[@]}" --out $O/report/held18 > /dev/null; rc1=$?
  $P experiments/route_follow/scripts/rf_follow.py --runs "${F[@]}" --out $O/report/follow --workers 16 --pairs "${PF[@]}" > /dev/null; rc2=$?
  echo "held $rc0 held18 $rc1 follow $rc2"; exit $((rc0 + rc1 + rc2)) ;;
*) echo "rf_v2.sh build | ident | pilot a|b | train a|b <base> <seed> | report [tags]" >&2; exit 2 ;;
esac
