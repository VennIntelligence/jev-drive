#!/usr/bin/env bash
# PAIT80 pilot (plans/2026-10-13-turning-school-prereg.md): RPAIT80-L18-s0 (on1_PAIT80) and RPAITC80-L18-s0 (on1_PAITC80), each followed by
# the held 460 / 18-decision read; the public 1491 / 10-decision guard for both; then one report job (held.py, rf_follow, ol_confirm guard).
#   bash experiments/route_follow/scripts/rf_pait80.sh submit   -> job ids in $O/jobs.txt
#   bash experiments/route_follow/scripts/rf_pait80.sh report   (job body)
set -uo pipefail; cd "$(dirname "$0")/../../.."
RF=$DATA_DIR/runs/route_follow; O=$RF/pait80; G=$RF/guard; M=$DATA_DIR/runs/route_mem/read; E=experiments/onpol_long; W=$DATA_DIR/runs/onpol_long
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python
REC="--init RB0-F-s0 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4"
NEW="RPAIT80-L18-s0 RPAITC80-L18-s0"; REF="RSG-L18-s0 RNONEG-L18-s0 RB0-F-s0"
case $1 in
submit)
  mkdir -p $O/pool; sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner route-follow --name "$name" --tries 1 --priority 14 --no-check \
       --log-dir $O/pool/$name "$@" | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
  dep=()
  for x in RPAIT80:on1_PAIT80:"" RPAITC80:on1_PAITC80:"pool=a:9,"; do A=${x%%:*}; r=${x#*:}; pool=${r%%:*}; w=${r#*:}; T=$A-L18-s0
    t=$(sub $(echo $A | tr A-Z a-z)-s0 --train --vram 28 --cpu 6 --ram 70 --timeout-h 3 -- $TR experiments/route_mem/scripts/rm_train.py train --tag $T $REC \
        --on $pool --on-k 12 --on-mode both --on-weight "${w}topo=straight_exit:2,topo=straight_road:2" --seed 0) || exit 1
    PRIO=14 bash experiments/route_mem/scripts/rm_chain.sh submit-read $T $t || exit 1
    dep+=($(grep " rm-cover-$T$" $DATA_DIR/runs/route_mem/jobs.txt | tail -1 | cut -d' ' -f1))
    for i in $(seq 0 9); do
      dep+=($(sub pait80g-$A-$i --after $t --vram 27 --cpu 8 --ram 30 --timeout-h 2 --gpus 1,2,3,4,5 --pin-strict --env CL_HIST_KEY=ol-fast-read -- bash $E/scripts/ol_chain.sh readlist $T \
            $W/lists/all_n10/c$i.txt 10 $G/read10/$T/b$i)) || exit 1
    done
  done
  sub pait80-report --after "$(IFS=,; echo "${dep[*]}")" --vram 0.5 --cpu 16 --ram 32 --timeout-h 1 -- bash experiments/route_follow/scripts/rf_pait80.sh report > /dev/null || exit 1
  cat $O/jobs.txt ;;
report)
  mkdir -p $O; R=""; for t in $NEW $REF RSG-L18-s1 RNONEG-L18-s1; do R="$R $t@18=$(ls -d $M/$t/c*/run | paste -sd,)"; done
  python3 $RF/held.py $(echo $R | sed 's/@18=/=/g') > $O/held_raw.txt; rc0=$?
  PR=(); for n in $NEW; do PR+=($n@18:RSG-L18-s0@18 $n@18:RNONEG-L18-s0@18 $n@18:RB0-F-s0@18); done; PR+=(RPAITC80-L18-s0@18:RPAIT80-L18-s0@18)
  $P experiments/route_follow/scripts/rf_follow.py --runs $R --out $O/follow --workers 32 --pairs "${PR[@]}"; rc1=$?
  A=$DATA_DIR/runs/alpasim; Q=()
  Q+=("YR10m10-F-s0@all10=$(python3 -c "import json; a = json.load(open('$A/ot3/a/manifest.json')); b = json.load(open('$A/cf1/a/manifest.json')); print(','.join(a['YR10m10-F-s0'] + a['YR10m10-F-s0@new'] + b['YR10m10-F-s0@fresh']))")")
  for T in RSG-L18-s0 RSG-L18-s1 RCFT-L18-s0 $NEW; do Q+=("$T@all10=$(ls -d $G/read10/$T/b*/ | sed 's#/$##' | paste -sd,)"); done
  $P $E/scripts/ol_confirm.py --runs "${Q[@]}" --base YR10m10-F-s0 --mean RSG-L18=RSG-L18-s0,RSG-L18-s1 \
     --pairs RPAIT80-L18-s0:RSG-L18-s0 RPAITC80-L18-s0:RSG-L18-s0 RPAIT80-L18-s0:RCFT-L18-s0 --out $O/guard; rc2=$?
  echo "held $rc0 follow $rc1 guard $rc2"; exit $((rc0 + rc1 + rc2)) ;;
*) echo "rf_pait80.sh submit | report" >&2; exit 2 ;;
esac
