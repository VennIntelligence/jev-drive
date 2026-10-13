#!/usr/bin/env bash
# STOPLINE (plans/2026-10-13-stopline-prereg.md): RSGB-L18-s{0,1} = RSG-L18's command + --on-lon on_lon (pack items B, P of on1_L18RB0), each
# followed (--after) by the public 1491 / 10-decision read (10 chunks, ol_chain.sh readlist, rf_allrows.sh's way); then one confirm job
# (ol_confirm.py vs the stored YR10m10-F-s0 rollouts, RSG-L18's guard reads as the reference) and the held offline read (rf_lonread.py).
#   bash experiments/route_follow/scripts/rf_stopline.sh submit      -> job ids in $O/jobs.txt
#   bash experiments/route_follow/scripts/rf_stopline.sh confirm     (job body)
set -uo pipefail; cd "$(dirname "$0")/../../.."
E=experiments/onpol_long; W=$DATA_DIR/runs/onpol_long; G=$DATA_DIR/runs/route_follow/guard; O=$DATA_DIR/runs/route_follow/stopline
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python
TAGS="RSGB-L18-s0 RSGB-L18-s1"
REC="--init RB0-F-s0 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-k 13 --on-split alpasim_nuplan/onpol-train --on-mode self --on on1_L18RB0 --on-lon on_lon"
dirs() { ls -d $1/$2/b*/ | sed 's#/$##' | paste -sd,; }
case $1 in
submit)
  mkdir -p $O; sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner stopline --name "$name" --tries 1 --priority 12 --no-check \
       --log-dir $O/pool/$name "$@" | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
  rr=(); tt=()
  for T in $TAGS; do
    t=$(sub tr-$T --train --vram 28 --cpu 6 --ram 70 --timeout-h 2 -- $TR experiments/route_mem/scripts/rm_train.py train --tag $T $REC --seed ${T##*-s}) || exit 1
    tt+=($t)
    for i in $(seq 0 9); do
      rr+=($(sub sl10-$T-$i --after $t --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 -- bash $E/scripts/ol_chain.sh readlist $T $W/lists/all_n10/c$i.txt 10 $O/read10/$T/b$i)) || exit 1
    done
  done
  sub sl10-confirm --after "$(IFS=,; echo "${rr[*]}")" --vram 0.5 --cpu 16 --ram 32 --timeout-h 1 -- bash experiments/route_follow/scripts/rf_stopline.sh confirm > /dev/null || exit 1
  sub sl-lonread --after "$(IFS=,; echo "${tt[*]}")" --vram 12 --cpu 6 --ram 60 --timeout-h 1 -- $TR experiments/route_follow/scripts/rf_lonread.py --pool on1_L18RSGallheld \
      --tags RSG-L18-s0 RSG-L18-s1 RB0-F-s0 $TAGS --out $O/lonread > /dev/null || exit 1
  cat $O/jobs.txt ;;
confirm)
  A=$DATA_DIR/runs/alpasim; R=()
  R+=("YR10m10-F-s0@all10=$(python3 -c "import json; a = json.load(open('$A/ot3/a/manifest.json')); b = json.load(open('$A/cf1/a/manifest.json')); print(','.join(a['YR10m10-F-s0'] + a['YR10m10-F-s0@new'] + b['YR10m10-F-s0@fresh']))")")
  for T in RSG-L18-s0 RSG-L18-s1; do R+=("$T@all10=$(dirs $G/read10 $T)"); done
  for T in $TAGS; do R+=("$T@all10=$(dirs $O/read10 $T)"); done
  $P $E/scripts/ol_confirm.py --runs "${R[@]}" --base YR10m10-F-s0 --mean RSG-L18=RSG-L18-s0,RSG-L18-s1 RSGB-L18=RSGB-L18-s0,RSGB-L18-s1 \
     --pairs RSGB-L18:RSG-L18 RSGB-L18-s0:RSG-L18 RSGB-L18-s1:RSG-L18 --out $O/confirm ;;
*) echo "rf_stopline.sh submit | confirm" >&2; exit 2 ;;
esac
