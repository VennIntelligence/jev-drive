#!/usr/bin/env bash
# ROUTE v2 (plans/2026-10-15-route-v2-prereg.md): ordered route input (--rm-order), near-field lane hinge with negatives and the over-turn
# guard (--lane-lam), lane-change counterfactual pairs (--lc-cf) on top of RSG-L18-s0's exact recipe (route_mem/train-RSG-L18-s0 meta.json).
# Every GPU job through the pool at low priority; ids appended to $O/jobs.txt. Box, repo root.
#   rf_v2.sh build                      CPU: ordered feature files (rm_build.py order) of the 12 navtrain shards + yaw-rate rows, pool
#                                       on1_L18RB0 and the counterfactual pools on1_CF / on1_CFdev; then the gap check (rm_build cover-like
#                                       summary of the training rows vs the served held reads of RSG-L18-s0)
#   rf_v2.sh ident                      bit identity with every flag off: 50 eager steps of RSG-L18-s0's command with this code and with the
#                                       code before route v2 (git worktree at $BASE_REV), then on_train ident + the dev ADE of both logs
#   rf_v2.sh pilot [flags]              RV2P-L18-s0: 1 000 steps, seed 0, batch 64 (on-k 7, lc-k 4), gentle schedule, init RSG-L18-s0, flags
#                                       (default: all three), then rf_v2gate.py vs RSG-L18-s0
#   rf_v2.sh train <base> <seed> [flags]   RV2-L18-s<seed> (TAG_PREFIX overrides RV2): 2 500 gentle steps on <base> with the flags (default all
#                                       three), then the held 460 / 18-decision read (rm_chain.sh read, 6 chunks); read ids -> $O/reads.txt
#   rf_v2.sh report [tags]              one pool job after every read in $O/reads.txt: held.py, ol_confirm.py paired vs the base(s) and RSG-L18-s0,
#                                       rf_follow.py turn completion -> $O/report/
#   rf_v2.sh body-train <tag> <init> <seed> <steps> <batch> <on-k> [flags] | body-report [tags]   (job bodies)
set -uo pipefail; cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/route_follow/v2; M=$DATA_DIR/runs/route_mem/read; CK=$DATA_DIR/runs/op_parity/runs
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; ME=experiments/route_follow/scripts/rf_v2.sh
PRIO=${PRIO:-6}; BASE_REV=${BASE_REV:-e58d65bf}
# RSG-L18-s0's command (meta.json argv) without --init / --steps / --seed / --tag
REC="--warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-split alpasim_nuplan/onpol-train --on-mode self --on on1_L18RB0"
ALL_FLAGS="--rm-order --lane-lam 10 --lane-dz 0.5 --lane-neg 0.2 --lc-cf"
D12=$(for i in $(seq 0 11); do printf "navtrain_full.s%dof12 " $i; done)
mkdir -p $O/pool
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner route-follow --name "$name" --tries 1 --priority $PRIO --no-check \
     --log-dir $O/pool/$name "$@" | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
lcflags() { local k=$1; shift; local f="$*"; [[ $f == *--lc-cf* && $f != *--lc-k* ]] && f="$f --lc-k $k"; echo "$f"; }
case $1 in
build)
  sub v2-build --vram 0 --cpu 24 --ram 60 --timeout-h 1 -- $P experiments/route_mem/scripts/rm_build.py order --data $D12 --pool L18RB0 --cf on1_CF on1_CFdev --workers 24 ;;
ident)
  [[ -d /tmp/jev-v2base ]] || git worktree add --detach /tmp/jev-v2base $BASE_REV > /dev/null || exit 1
  for t in RV2ID-new-s0 RV2ID-old-s0; do [[ ! -e $CK/$t ]] || { echo "$CK/$t exists"; exit 1; }; done
  a=$(sub v2-ident-new --train --vram 28 --cpu 6 --ram 70 --timeout-h 1 -- bash $ME body-train RV2ID-new-s0 RSG-L18-s0 0 50 128 13 --eval-every 50 --allow-init) || exit 1
  b=$(sub v2-ident-old --train --vram 28 --cpu 6 --ram 70 --timeout-h 1 --cwd /tmp/jev-v2base -- $TR experiments/route_mem/scripts/rm_train.py train --tag RV2ID-old-s0 \
      --init RSG-L18-s0 --steps 50 --batch 128 --on-k 13 --seed 0 $REC --eval-every 50 --allow-init) || exit 1
  sub v2-ident-cmp --after $a,$b --vram 0 --cpu 1 --ram 8 --timeout-h 1 -- bash -c "$TR experiments/onpol_nuplan/scripts/on_train.py ident --a RV2ID-new-s0 --b RV2ID-old-s0 --out $O/ident.json; rc=\$?; \
      grep -h 'dev @' $DATA_DIR/runs/route_mem/train-RV2ID-new-s0/*/log.txt $DATA_DIR/runs/route_mem/train-RV2ID-old-s0/*/log.txt | tee $O/ident_dev.txt; exit \$rc" ;;
pilot)
  shift; F="${*:-$ALL_FLAGS}"; T=RV2P-L18-s0
  t=$(sub v2-pilot --train --vram 24 --cpu 6 --ram 70 --timeout-h 1 -- bash $ME body-train $T RSG-L18-s0 0 1000 64 7 $(lcflags 4 $F) --allow-init) || exit 1
  sub v2-gate-$T --after $t --vram 20 --cpu 4 --ram 48 --timeout-h 1 -- $TR experiments/route_follow/scripts/rf_v2gate.py --tag $T --base RSG-L18-s0 > /dev/null || exit 1
  tail -2 $O/jobs.txt ;;
train)
  B=$2; s=$3; shift 3; F="${*:-$ALL_FLAGS}"; T=${TAG_PREFIX:-RV2}-L18-s$s
  [[ -f $CK/$B/ckpt-final.pt ]] || { echo "no base $B"; exit 1; }
  t=$(sub v2-train-$T --train --vram 28 --cpu 6 --ram 70 --timeout-h 2 -- bash $ME body-train $T $B $s 2500 128 13 $(lcflags 8 $F) --allow-init) || exit 1
  sub v2-gate-$T --after $t --vram 20 --cpu 4 --ram 48 --timeout-h 1 -- $TR experiments/route_follow/scripts/rf_v2gate.py --tag $T --base $B > /dev/null || exit 1
  for i in 0 1 2 3 4 5; do
    sub v2-read-$T-$i --after $t --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
        -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6 >> $O/reads.txt || exit 1
  done
  echo "$T $B" >> $O/bases.txt; tail -8 $O/jobs.txt ;;
report)
  shift; sub v2-report --after "$(paste -sd, $O/reads.txt)" --vram 0.5 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME body-report "$@" > /dev/null || exit 1; tail -1 $O/jobs.txt ;;
body-train)
  T=$2; I=$3; s=$4; N=$5; BS=$6; K=$7; shift 7
  exec $TR experiments/route_mem/scripts/rm_train.py train --tag $T --init $I --steps $N --batch $BS --on-k $K --seed $s $REC "$@" ;;
body-report)
  shift; ALL="${*:-$(cut -d' ' -f1 $O/bases.txt | awk '!s[$0]++' | paste -sd' ')}"
  BASES=$(for t in $ALL; do grep "^$t " $O/bases.txt | tail -1 | cut -d' ' -f2; done | awk '!s[$0]++' | paste -sd' ')
  REF=$(echo "$BASES RSG-L18-s0 RSG-L18-s1" | tr ' ' '\n' | awk 'NF && !s[$0]++' | paste -sd' ')
  ALL=$(for t in $ALL $REF; do ls $M/$t/c*/run/aggregate/results-summary.json > /dev/null 2>&1 && echo $t; done | awk '!s[$0]++' | paste -sd' ')
  mkdir -p $O/report; R=""; H=(); F=(); PH=(); PF=()
  for t in $ALL; do d=$(ls -d $M/$t/c*/run | paste -sd,); R="$R $t=$d"; H+=("$t@held18=$d"); F+=("$t@18=$d"); done
  for t in ${*:-$(cut -d' ' -f1 $O/bases.txt | awk '!s[$0]++')}; do b=$(grep "^$t " $O/bases.txt | tail -1 | cut -d' ' -f2)
    for r in $b RSG-L18-s${t##*-s}; do [[ " $ALL " == *" $r "* && $r != $t ]] && { PH+=("$t:$r"); PF+=("$t@18:$r@18"); }; done; done
  python3 $DATA_DIR/runs/route_follow/held.py $R > $O/report/held_raw.txt; rc0=$?
  $P experiments/onpol_long/scripts/ol_confirm.py --runs "${H[@]}" --base RSG-L18-s0 --pairs "${PH[@]}" --out $O/report/held18 > /dev/null; rc1=$?
  $P experiments/route_follow/scripts/rf_follow.py --runs "${F[@]}" --out $O/report/follow --workers 16 --pairs "${PF[@]}" > /dev/null; rc2=$?
  echo "held $rc0 held18 $rc1 follow $rc2"; exit $((rc0 + rc1 + rc2)) ;;
*) echo "rf_v2.sh build | ident | pilot [flags] | train <base> <seed> [flags] | report [tags]" >&2; exit 2 ;;
esac
