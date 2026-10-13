#!/usr/bin/env bash
# RED (plans/2026-10-13-red-prereg.md): RRED-L18-s{0,1} on on1_RED (B only, imitation of the stop / log targets), each followed (--after) by the
# held 460 / 18-decision read (rm_chain.sh submit-read, 6 jobs + coverage); the public 1491 / 10-decision guard for s0 (10 ol_chain readlist
# chunks, CFT's way); line (a) on on1_REDheld (rf_redread.py read); then one report job (held.py, ol_confirm on held 18 and all10).
#   bash experiments/route_follow/scripts/rf_red.sh submit     -> job ids in $O/jobs.txt
#   bash experiments/route_follow/scripts/rf_red.sh report     (job body)
set -uo pipefail; cd "$(dirname "$0")/../../.."
RF=$DATA_DIR/runs/route_follow; O=$RF/red; G=$RF/guard; M=$DATA_DIR/runs/route_mem/read; E=experiments/onpol_long; W=$DATA_DIR/runs/onpol_long
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python
REC="--init RB0-F-s0 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on on1_RED --on-k 14 --on-mode both --on-lon on_lon"
HELD="RRED-L18-s0 RRED-L18-s1 RNONEG-L18-s0 RNONEG-L18-s1 RSG-L18-s0 RSG-L18-s1 RB0-F-s0 RB0-F-s1"
case $1 in
submit)
  mkdir -p $O/pool; sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner red --name "$name" --tries 1 --priority 14 --no-check \
       --log-dir $O/pool/$name "$@" | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
  tt=(); dep=()
  for s in 0 1; do T=RRED-L18-s$s
    t=$(sub red-tr-s$s --train --vram 28 --cpu 6 --ram 70 --timeout-h 2 -- $TR experiments/route_mem/scripts/rm_train.py train --tag $T $REC --seed $s) || exit 1
    tt+=($t)
    PRIO=14 bash experiments/route_mem/scripts/rm_chain.sh submit-read $T $t || exit 1
    dep+=($(grep " rm-cover-$T$" $DATA_DIR/runs/route_mem/jobs.txt | tail -1 | cut -d' ' -f1))
  done
  for i in $(seq 0 9); do
    dep+=($(sub redg-rg10-$i --after ${tt[0]} --vram 20 --cpu 6 --ram 30 --timeout-h 2 --gpus 1,2,3,4,5 --pin-strict --env CL_HIST_KEY=alpasim-read4 -- bash $E/scripts/ol_chain.sh readlist RRED-L18-s0 \
          $W/lists/all_n10/c$i.txt 10 $G/read10/RRED-L18-s0/b$i)) || exit 1
  done
  dep+=($(sub red-read --after "$(IFS=,; echo "${tt[*]}")" --vram 16 --cpu 4 --ram 50 --timeout-h 1 -- $TR experiments/route_follow/scripts/rf_redread.py read \
        --tags RRED-L18-s0 RRED-L18-s1 RNONEG-L18-s0 RNONEG-L18-s1 RSG-L18-s0 RSG-L18-s1 RB0-F-s0 --groups RRED=RRED-L18-s0,RRED-L18-s1 \
        RNONEG=RNONEG-L18-s0,RNONEG-L18-s1 RSG=RSG-L18-s0,RSG-L18-s1 --pairs RRED:RNONEG RRED:RSG RRED-L18-s0:RNONEG-L18-s0 RRED-L18-s1:RNONEG-L18-s1 \
        --out $O/read)) || exit 1
  sub red-report --after "$(IFS=,; echo "${dep[*]}")" --vram 0.5 --cpu 16 --ram 32 --timeout-h 1 -- bash experiments/route_follow/scripts/rf_red.sh report > /dev/null || exit 1
  cat $O/jobs.txt ;;
report)
  R=""; for t in $HELD; do R="$R $t=$(ls -d $M/$t/c*/run | paste -sd,)"; done
  python3 $RF/held.py $R > $O/held_raw.txt; rc0=$?
  H=(); for t in $HELD; do H+=("$t@held18=$(ls -d $M/$t/c*/run | paste -sd,)"); done
  $P $E/scripts/ol_confirm.py --runs "${H[@]}" --base RB0-F-s0 --mean RRED-L18=RRED-L18-s0,RRED-L18-s1 RNONEG-L18=RNONEG-L18-s0,RNONEG-L18-s1 \
     RSG-L18=RSG-L18-s0,RSG-L18-s1 RB0-F=RB0-F-s0,RB0-F-s1 --pairs RRED-L18:RSG-L18 RRED-L18:RNONEG-L18 RRED-L18-s0:RSG-L18-s0 RRED-L18-s1:RSG-L18-s0 --out $O/held18; rc1=$?
  A=$DATA_DIR/runs/alpasim; Q=()
  Q+=("YR10m10-F-s0@all10=$(python3 -c "import json; a = json.load(open('$A/ot3/a/manifest.json')); b = json.load(open('$A/cf1/a/manifest.json')); print(','.join(a['YR10m10-F-s0'] + a['YR10m10-F-s0@new'] + b['YR10m10-F-s0@fresh']))")")
  for T in RSG-L18-s0 RSG-L18-s1 RRED-L18-s0; do Q+=("$T@all10=$(ls -d $G/read10/$T/b*/ | sed 's#/$##' | paste -sd,)"); done
  $P $E/scripts/ol_confirm.py --runs "${Q[@]}" --base YR10m10-F-s0 --mean RSG-L18=RSG-L18-s0,RSG-L18-s1 \
     --pairs RRED-L18-s0:RSG-L18 RRED-L18-s0:RSG-L18-s0 --out $O/guard; rc2=$?
  echo "held $rc0 held18 $rc1 guard $rc2"; exit $((rc0 + rc1 + rc2)) ;;
*) echo "rf_red.sh submit | report" >&2; exit 2 ;;
esac
