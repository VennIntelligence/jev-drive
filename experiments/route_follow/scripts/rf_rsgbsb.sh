#!/usr/bin/env bash
# RSGbothSB (side-bias fix 1, plans/2026-10-13-rsgboth-prereg.md Amendment 2): RSGboth-L18-s{k}'s exact command plus
#   --on on1_L18RB0s --on-weight rside=R_in:2,rside=R_out:0.48
# (rf_sidetag.py pool: the same rows and targets, tags tside / pside / rside added; right-turn rows outside : inside the human path drawn 1 : 1
# instead of 321 : 77). Each train job (60-step preflight smoke) is followed (--after) by the held 460 / 18-decision read (rm_chain.sh
# submit-read) and the public 1491 / 10-decision guard (10 ol_chain readlist chunks); one report job at the end: held.py, ol_confirm held18 +
# guard, rf_follow, rf_inside, rf_tax extract + rf_sidebias held of the new tags. Box, repo root.   rf_rsgbsb.sh submit | report
set -uo pipefail; cd "$(dirname "$0")/../../.."
RF=$DATA_DIR/runs/route_follow; O=$RF/rsgboth_sb; G=$RF/guard; M=$DATA_DIR/runs/route_mem/read; E=experiments/onpol_long; W=$DATA_DIR/runs/onpol_long
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python
REC="--init RB0-F-s0 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-k 13 --on-split alpasim_nuplan/onpol-train --on-mode both"
SB="--on on1_L18RB0s --on-weight rside=R_in:2,rside=R_out:0.48"
NEW="RSGbothSB-L18-s0 RSGbothSB-L18-s1"
REF="RSGboth-L18-s0 RSGboth-L18-s1 RSG-L18-s0 RSG-L18-s1"
case $1 in
submit)
  mkdir -p $O/pool; sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner route-follow --name "$name" --tries 1 --priority 14 --no-check \
       --log-dir $O/pool/$name "$@" | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
  dep=()
  for T in $NEW; do s=${T##*-s}
    t=$(sub rsgbsb-s$s --train --gpus 1,2,3,4,5 --pin-strict --vram 28 --cpu 6 --ram 70 --timeout-h 3 \
        --preflight "$TR experiments/route_mem/scripts/rm_train.py train --tag RSGbothSB-smoke-s$s $REC $SB --seed $s --steps 60 --eval-every 60" \
        -- $TR experiments/route_mem/scripts/rm_train.py train --tag $T $REC $SB --seed $s) || exit 1
    PRIO=14 bash experiments/route_mem/scripts/rm_chain.sh submit-read $T $t || exit 1
    dep+=($(grep " rm-cover-$T$" $DATA_DIR/runs/route_mem/jobs.txt | tail -1 | cut -d' ' -f1))
    for i in $(seq 0 9); do
      dep+=($(sub rsgbsb-g10-$T-$i --after $t --vram 27 --cpu 8 --ram 30 --timeout-h 2 --gpus 1,2,3,4,5 --pin-strict --env CL_HIST_KEY=ol-fast-read -- bash $E/scripts/ol_chain.sh readlist $T \
            $W/lists/all_n10/c$i.txt 10 $G/read10/$T/b$i)) || exit 1
    done
  done
  sub rsgbsb-report --after "$(IFS=,; echo "${dep[*]}")" --vram 0.5 --cpu 16 --ram 48 --timeout-h 2 -- bash experiments/route_follow/scripts/rf_rsgbsb.sh report > /dev/null || exit 1
  cat $O/jobs.txt ;;
report)
  ALL="$NEW $REF"
  R=""; for t in $ALL; do R="$R $t=$(ls -d $M/$t/c*/run | paste -sd,)"; done
  python3 $RF/held.py $R > $O/held_raw.txt; rc0=$?
  H=(); F=(); for t in $ALL; do d=$(ls -d $M/$t/c*/run | paste -sd,); H+=("$t@held18=$d"); F+=("$t@18=$d"); done
  PH=(); PF=(); for s in 0 1; do for r in RSG-L18 RSGboth-L18; do PH+=("RSGbothSB-L18-s$s:$r-s$s"); PF+=("RSGbothSB-L18-s$s@18:$r-s$s@18"); done; done
  MN="RSGbothSB-L18=RSGbothSB-L18-s0,RSGbothSB-L18-s1 RSGboth-L18=RSGboth-L18-s0,RSGboth-L18-s1 RSG-L18=RSG-L18-s0,RSG-L18-s1"
  $P $E/scripts/ol_confirm.py --runs "${H[@]}" --base RSG-L18-s0 --mean $MN --pairs RSGbothSB-L18:RSG-L18 RSGbothSB-L18:RSGboth-L18 "${PH[@]}" --out $O/held18; rc1=$?
  $P experiments/route_follow/scripts/rf_follow.py --runs "${F[@]}" --out $O/follow --workers 16 --pairs "${PF[@]}"; rc2=$?
  $P experiments/route_follow/scripts/rf_inside.py --follow $O/follow --pairs "${PF[@]}" --out $O/inside.json > /dev/null; rc3=$?
  A=$DATA_DIR/runs/alpasim; Q=()
  Q+=("YR10m10-F-s0@all10=$(python3 -c "import json; a = json.load(open('$A/ot3/a/manifest.json')); b = json.load(open('$A/cf1/a/manifest.json')); print(','.join(a['YR10m10-F-s0'] + a['YR10m10-F-s0@new'] + b['YR10m10-F-s0@fresh']))")")
  for T in $ALL; do Q+=("$T@all10=$(ls -d $G/read10/$T/b*/ | sed 's#/$##' | paste -sd,)"); done
  $P $E/scripts/ol_confirm.py --runs "${Q[@]}" --base YR10m10-F-s0 --mean $MN --pairs RSGbothSB-L18:RSG-L18 RSGbothSB-L18:RSGboth-L18 "${PH[@]}" --out $O/guard; rc4=$?
  rc5=0; for T in $NEW; do
    $P experiments/route_follow/scripts/rf_tax.py extract --runs $(ls -d $M/$T/c*/run) --out $O/held_$T.pkl --workers 8 > $O/extract_$T.log 2>&1 || rc5=1; done
  S=(); for T in $NEW; do S+=("$T=$O/held_$T.pkl"); done
  .venv/bin/python experiments/route_follow/scripts/rf_sidebias.py held --set "${S[@]}" --out $O/side --workers 8; rc6=$?
  echo "held $rc0 held18 $rc1 follow $rc2 inside $rc3 guard $rc4 extract $rc5 side $rc6"; exit $((rc0 + rc1 + rc2 + rc3 + rc4 + rc5 + rc6)) ;;
*) echo "rf_rsgbsb.sh submit | report" >&2; exit 2 ;;
esac
