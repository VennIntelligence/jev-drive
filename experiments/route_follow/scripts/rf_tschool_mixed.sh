#!/usr/bin/env bash
# Turning school, Amendment 4: RSGmixPAIT-L18-s{0,1} = RSGPAIT-L18's exact command (pool on1_RSGPAIT, PAI rows ~30 % of draws) with --on-mode mixed
# (S rows `self` = hinges only as RSG-L18; PAI rows `both`). Each followed by the held 460 / 18-decision read (rm_chain.sh submit-read) and the public
# 1491 / 10-decision guard; one report job (held18 confirm, rf_follow turn completion, rf_inside, guard confirm). Box, repo root.
#   rf_tschool_mixed.sh submit | report
set -uo pipefail; cd "$(dirname "$0")/../../.."
RF=$DATA_DIR/runs/route_follow; O=$RF/tschool; G=$RF/guard; M=$DATA_DIR/runs/route_mem/read; E=experiments/onpol_long; W=$DATA_DIR/runs/onpol_long
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; OUT=$O/mixed
REC="--init RB0-F-s0 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4"
ARM="--on on1_RSGPAIT --on-k 13 --on-split alpasim_nuplan/onpol-train --on-mode mixed --on-weight pool=p:1.0306,topo=straight_exit:2,topo=straight_road:2"
NEW="RSGmixPAIT-L18-s0 RSGmixPAIT-L18-s1"
HELD="$NEW RSGPAIT-L18-s0 RSGPAIT-L18-s1 RSGboth-L18-s0 RSGboth-L18-s1 RSG-L18-s0 RSG-L18-s1 RB0-F-s0 RB0-F-s1"
case $1 in
submit)
  mkdir -p $OUT/pool; sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner tschool --name "$name" --tries 1 --priority 14 --no-check \
       --log-dir $OUT/pool/$name "$@" | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $OUT/jobs.txt; echo "$id"; }
  dep=()
  for T in $NEW; do s=${T##*-s}
    t=$(sub tsm-tr-$T --train --vram 28 --cpu 6 --ram 70 --timeout-h 2 -- $TR experiments/route_mem/scripts/rm_train.py train --tag $T $REC $ARM --seed $s) || exit 1
    PRIO=14 bash experiments/route_mem/scripts/rm_chain.sh submit-read $T $t || exit 1
    dep+=($(grep " rm-cover-$T$" $DATA_DIR/runs/route_mem/jobs.txt | tail -1 | cut -d' ' -f1))
    for i in $(seq 0 9); do
      dep+=($(sub tsm-g10-$T-$i --after $t --vram 27 --cpu 8 --ram 30 --timeout-h 2 --gpus 1,2,3,4,5 --pin-strict --env CL_HIST_KEY=ol-fast-read -- bash $E/scripts/ol_chain.sh readlist $T \
            $W/lists/all_n10/c$i.txt 10 $G/read10/$T/b$i)) || exit 1
    done
  done
  sub tsm-report --after "$(IFS=,; echo "${dep[*]}")" --vram 0.5 --cpu 16 --ram 32 --timeout-h 1 -- bash experiments/route_follow/scripts/rf_tschool_mixed.sh report > /dev/null || exit 1
  cat $OUT/jobs.txt ;;
report)
  R=""; for t in $HELD; do R="$R $t=$(ls -d $M/$t/c*/run | paste -sd,)"; done
  python3 $RF/held.py $R > $OUT/held_raw.txt; rc0=$?
  H=(); F=(); for t in $HELD; do d=$(ls -d $M/$t/c*/run | paste -sd,); H+=("$t@held18=$d"); F+=("$t@18=$d"); done
  $P $E/scripts/ol_confirm.py --runs "${H[@]}" --base RB0-F-s0 --mean RSGmixPAIT-L18=RSGmixPAIT-L18-s0,RSGmixPAIT-L18-s1 RSGPAIT-L18=RSGPAIT-L18-s0,RSGPAIT-L18-s1 \
     RSGboth-L18=RSGboth-L18-s0,RSGboth-L18-s1 RSG-L18=RSG-L18-s0,RSG-L18-s1 RB0-F=RB0-F-s0,RB0-F-s1 \
     --pairs RSGmixPAIT-L18:RSG-L18 RSGmixPAIT-L18-s0:RSG-L18-s0 RSGmixPAIT-L18-s1:RSG-L18-s1 RSGmixPAIT-L18:RSGPAIT-L18 RSGmixPAIT-L18:RSGboth-L18 --out $OUT/held18; rc1=$?
  $P experiments/route_follow/scripts/rf_follow.py --runs "${F[@]}" --out $OUT/follow --workers 48 --pairs RSGmixPAIT-L18-s0@18:RSG-L18-s0@18 RSGmixPAIT-L18-s1@18:RSG-L18-s1@18 \
     RSGmixPAIT-L18-s0@18:RB0-F-s0@18 RSGmixPAIT-L18-s1@18:RB0-F-s0@18; rc2=$?
  $P experiments/route_follow/scripts/rf_inside.py --follow $OUT/follow --out $OUT/inside.json \
     --pairs RSGmixPAIT-L18-s0@18:RSG-L18-s0@18 RSGmixPAIT-L18-s1@18:RSG-L18-s1@18; rc3=$?
  A=$DATA_DIR/runs/alpasim; Q=()
  Q+=("YR10m10-F-s0@all10=$(python3 -c "import json; a = json.load(open('$A/ot3/a/manifest.json')); b = json.load(open('$A/cf1/a/manifest.json')); print(','.join(a['YR10m10-F-s0'] + a['YR10m10-F-s0@new'] + b['YR10m10-F-s0@fresh']))")")
  for T in RSG-L18-s0 RSG-L18-s1 RSGboth-L18-s0 RSGboth-L18-s1 RSGPAIT-L18-s0 RSGPAIT-L18-s1 $NEW; do Q+=("$T@all10=$(ls -d $G/read10/$T/b*/ | sed 's#/$##' | paste -sd,)"); done
  $P $E/scripts/ol_confirm.py --runs "${Q[@]}" --base YR10m10-F-s0 --mean RSG-L18=RSG-L18-s0,RSG-L18-s1 RSGPAIT-L18=RSGPAIT-L18-s0,RSGPAIT-L18-s1 \
     RSGboth-L18=RSGboth-L18-s0,RSGboth-L18-s1 RSGmixPAIT-L18=RSGmixPAIT-L18-s0,RSGmixPAIT-L18-s1 \
     --pairs RSGmixPAIT-L18:RSG-L18 RSGmixPAIT-L18-s0:RSG-L18-s0 RSGmixPAIT-L18-s1:RSG-L18-s1 RSGmixPAIT-L18:RSGPAIT-L18 --out $OUT/guard; rc4=$?
  echo "held $rc0 held18 $rc1 follow $rc2 inside $rc3 guard $rc4"; exit $((rc0 + rc1 + rc2 + rc3 + rc4)) ;;
*) echo "rf_tschool_mixed.sh submit | report" >&2; exit 2 ;;
esac
