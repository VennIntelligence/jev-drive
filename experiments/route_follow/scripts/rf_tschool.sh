#!/usr/bin/env bash
# Turning school (plans/2026-10-13-turning-school-prereg.md): the four registered arms RPAIT-L18-s{0,1} (on1_PAIT, primary) and RPAITC-L18-s{0,1}
# (on1_PAITC, secondary), each followed (--after) by the held 460 / 18-decision read (rm_chain.sh submit-read, 6 jobs + coverage) and the
# public 1491 / 10-decision guard (10 ol_chain readlist chunks, CFT / RED's way); then one report job (held.py, ol_confirm on held 18 and all10).
# Box, repo root; needs on1_PAIT and on1_PAITC (rf_pait_chain.sh puts them there and calls `submit`).
#   bash experiments/route_follow/scripts/rf_tschool.sh submit     -> job ids in $O/jobs.txt
#   bash experiments/route_follow/scripts/rf_tschool.sh report     (job body)
set -uo pipefail; cd "$(dirname "$0")/../../.."
RF=$DATA_DIR/runs/route_follow; O=$RF/tschool; G=$RF/guard; M=$DATA_DIR/runs/route_mem/read; E=experiments/onpol_long; W=$DATA_DIR/runs/onpol_long
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python
REC="--init RB0-F-s0 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4"
ARM_RPAIT="--on on1_PAIT --on-k 12 --on-mode both --on-weight topo=straight_exit:2,topo=straight_road:2"
ARM_RPAITC="--on on1_PAITC --on-k 12 --on-mode both --on-weight pool=a:9,topo=straight_exit:2,topo=straight_road:2"
NEW="RPAIT-L18-s0 RPAIT-L18-s1 RPAITC-L18-s0 RPAITC-L18-s1"
HELD="$NEW RNONEG-L18-s0 RNONEG-L18-s1 RSG-L18-s0 RSG-L18-s1 RB0-F-s0 RB0-F-s1"
case $1 in
submit)
  for p in on1_PAIT on1_PAITC; do [[ -f $DATA_DIR/runs/op_parity/cache/$p/tab.npz && -f $DATA_DIR/runs/route_mem/feat/$p.npz ]] || { echo "no pool $p" >&2; exit 1; }; done
  mkdir -p $O/pool; sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner tschool --name "$name" --tries 1 --priority 14 --no-check \
       --log-dir $O/pool/$name "$@" | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
  dep=()
  for T in $NEW; do s=${T##*-s}; a=${T%%-L18*}; v=ARM_$a
    t=$(sub ts-tr-$T --train --vram 28 --cpu 6 --ram 70 --timeout-h 2 -- $TR experiments/route_mem/scripts/rm_train.py train --tag $T $REC ${!v} --seed $s) || exit 1
    PRIO=14 bash experiments/route_mem/scripts/rm_chain.sh submit-read $T $t || exit 1
    dep+=($(grep " rm-cover-$T$" $DATA_DIR/runs/route_mem/jobs.txt | tail -1 | cut -d' ' -f1))
    for i in $(seq 0 9); do
      dep+=($(sub ts-g10-$T-$i --after $t --vram 27 --cpu 8 --ram 30 --timeout-h 2 --gpus 1,2,3,4,5 --pin-strict --env CL_HIST_KEY=ol-fast-read -- bash $E/scripts/ol_chain.sh readlist $T \
            $W/lists/all_n10/c$i.txt 10 $G/read10/$T/b$i)) || exit 1
    done
  done
  sub ts-report --after "$(IFS=,; echo "${dep[*]}")" --vram 0.5 --cpu 16 --ram 32 --timeout-h 1 -- bash experiments/route_follow/scripts/rf_tschool.sh report > /dev/null || exit 1
  cat $O/jobs.txt ;;
report)
  R=""; for t in $HELD; do R="$R $t=$(ls -d $M/$t/c*/run | paste -sd,)"; done
  python3 $RF/held.py $R > $O/held_raw.txt; rc0=$?
  H=(); for t in $HELD; do H+=("$t@held18=$(ls -d $M/$t/c*/run | paste -sd,)"); done
  $P $E/scripts/ol_confirm.py --runs "${H[@]}" --base RB0-F-s0 --mean RPAIT-L18=RPAIT-L18-s0,RPAIT-L18-s1 RPAITC-L18=RPAITC-L18-s0,RPAITC-L18-s1 \
     RNONEG-L18=RNONEG-L18-s0,RNONEG-L18-s1 RSG-L18=RSG-L18-s0,RSG-L18-s1 RB0-F=RB0-F-s0,RB0-F-s1 \
     --pairs RPAIT-L18:RSG-L18 RPAIT-L18:RNONEG-L18 RPAITC-L18:RSG-L18 RPAITC-L18:RPAIT-L18 --out $O/held18; rc1=$?
  A=$DATA_DIR/runs/alpasim; Q=()
  Q+=("YR10m10-F-s0@all10=$(python3 -c "import json; a = json.load(open('$A/ot3/a/manifest.json')); b = json.load(open('$A/cf1/a/manifest.json')); print(','.join(a['YR10m10-F-s0'] + a['YR10m10-F-s0@new'] + b['YR10m10-F-s0@fresh']))")")
  for T in RSG-L18-s0 RSG-L18-s1 $NEW; do Q+=("$T@all10=$(ls -d $G/read10/$T/b*/ | sed 's#/$##' | paste -sd,)"); done
  $P $E/scripts/ol_confirm.py --runs "${Q[@]}" --base YR10m10-F-s0 --mean RSG-L18=RSG-L18-s0,RSG-L18-s1 RPAIT-L18=RPAIT-L18-s0,RPAIT-L18-s1 \
     RPAITC-L18=RPAITC-L18-s0,RPAITC-L18-s1 --pairs RPAIT-L18:RSG-L18 RPAITC-L18:RSG-L18 RPAIT-L18-s0:RSG-L18-s0 RPAIT-L18-s1:RSG-L18-s1 --out $O/guard; rc2=$?
  echo "held $rc0 held18 $rc1 guard $rc2"; exit $((rc0 + rc1 + rc2)) ;;
*) echo "rf_tschool.sh submit | report" >&2; exit 2 ;;
esac
