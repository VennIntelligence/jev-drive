#!/usr/bin/env bash
# RSGBOTH (loss-audit fix, research/route-follow/loss-audit.md section 4): RSGboth-L18-s{0,1} = RSG-L18's exact command with --on-mode self -> both.
# Each is followed (--after) by the held 460 / 18-decision read (rm_chain.sh submit-read), the G-swap (rf_gates.py) and the public 1491 / 10-decision
# guard (10 ol_chain readlist chunks); one report job at the end. Box, repo root.   rf_rsgboth.sh submit | report
set -uo pipefail; cd "$(dirname "$0")/../../.."
RF=$DATA_DIR/runs/route_follow; O=$RF/rsgboth; G=$RF/guard; M=$DATA_DIR/runs/route_mem/read; E=experiments/onpol_long; W=$DATA_DIR/runs/onpol_long
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python
REC="--init RB0-F-s0 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-k 13 --on-split alpasim_nuplan/onpol-train --on-mode both --on on1_L18RB0"
NEW="RSGboth-L18-s0 RSGboth-L18-s1"
HELD="$NEW RSG-L18-s0 RSG-L18-s1 RB0-F-s0 RB0-F-s1"
case $1 in
submit)
  mkdir -p $O/pool; sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner route-follow --name "$name" --tries 1 --priority 14 --no-check \
       --log-dir $O/pool/$name "$@" | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
  dep=()
  for T in $NEW; do s=${T##*-s}
    t=$(sub rsgboth-s$s --train --vram 28 --cpu 6 --ram 70 --timeout-h 3 -- $TR experiments/route_mem/scripts/rm_train.py train --tag $T $REC --seed $s) || exit 1
    PRIO=14 bash experiments/route_mem/scripts/rm_chain.sh submit-read $T $t || exit 1
    dep+=($(grep " rm-cover-$T$" $DATA_DIR/runs/route_mem/jobs.txt | tail -1 | cut -d' ' -f1))
    dep+=($(sub rsgboth-gswap-s$s --after $t --vram 20 --cpu 4 --ram 40 --timeout-h 1 -- $TR experiments/route_follow/scripts/rf_gates.py --tag $T --out $O/gswap_$T.json)) || exit 1
    for i in $(seq 0 9); do
      dep+=($(sub rsgboth-g10-$T-$i --after $t --vram 27 --cpu 8 --ram 30 --timeout-h 2 --gpus 1,2,3,4,5 --pin-strict --env CL_HIST_KEY=ol-fast-read -- bash $E/scripts/ol_chain.sh readlist $T \
            $W/lists/all_n10/c$i.txt 10 $G/read10/$T/b$i)) || exit 1
    done
  done
  sub rsgboth-report --after "$(IFS=,; echo "${dep[*]}")" --vram 0.5 --cpu 16 --ram 32 --timeout-h 1 -- bash experiments/route_follow/scripts/rf_rsgboth.sh report > /dev/null || exit 1
  cat $O/jobs.txt ;;
report)
  R=""; for t in $HELD; do R="$R $t=$(ls -d $M/$t/c*/run | paste -sd,)"; done
  python3 $RF/held.py $R > $O/held_raw.txt; rc0=$?
  H=(); F=(); for t in $HELD; do d=$(ls -d $M/$t/c*/run | paste -sd,); H+=("$t@held18=$d"); F+=("$t@18=$d"); done
  $P $E/scripts/ol_confirm.py --runs "${H[@]}" --base RB0-F-s0 --mean RSGboth-L18=RSGboth-L18-s0,RSGboth-L18-s1 RSG-L18=RSG-L18-s0,RSG-L18-s1 RB0-F=RB0-F-s0,RB0-F-s1 \
     --pairs RSGboth-L18:RSG-L18 RSGboth-L18-s0:RSG-L18-s0 RSGboth-L18-s1:RSG-L18-s1 --out $O/held18; rc1=$?
  $P experiments/route_follow/scripts/rf_follow.py --runs "${F[@]}" --out $O/follow --workers 48 --pairs RSGboth-L18-s0@18:RSG-L18-s0@18 RSGboth-L18-s1@18:RSG-L18-s1@18 \
     RSGboth-L18-s0@18:RB0-F-s0@18 RSGboth-L18-s1@18:RB0-F-s0@18; rc2=$?
  A=$DATA_DIR/runs/alpasim; Q=()
  Q+=("YR10m10-F-s0@all10=$(python3 -c "import json; a = json.load(open('$A/ot3/a/manifest.json')); b = json.load(open('$A/cf1/a/manifest.json')); print(','.join(a['YR10m10-F-s0'] + a['YR10m10-F-s0@new'] + b['YR10m10-F-s0@fresh']))")")
  for T in YS-L18-s0 YS-L18-s1; do Q+=("$T@all10=$(ls -d $W/read10/$T/[ab]*/ | sed 's#/$##' | paste -sd,)"); done
  for T in RSG-L18-s0 RSG-L18-s1 $NEW; do Q+=("$T@all10=$(ls -d $G/read10/$T/b*/ | sed 's#/$##' | paste -sd,)"); done
  $P $E/scripts/ol_confirm.py --runs "${Q[@]}" --base YR10m10-F-s0 --mean YS-L18=YS-L18-s0,YS-L18-s1 RSG-L18=RSG-L18-s0,RSG-L18-s1 RSGboth-L18=RSGboth-L18-s0,RSGboth-L18-s1 \
     --pairs RSGboth-L18:RSG-L18 RSGboth-L18-s0:RSG-L18-s0 RSGboth-L18-s1:RSG-L18-s1 RSGboth-L18:YS-L18 --out $O/guard; rc3=$?
  echo "held $rc0 held18 $rc1 follow $rc2 guard $rc3"; exit $((rc0 + rc1 + rc2 + rc3)) ;;
*) echo "rf_rsgboth.sh submit | report" >&2; exit 2 ;;
esac
