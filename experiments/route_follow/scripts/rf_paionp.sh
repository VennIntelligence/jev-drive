#!/usr/bin/env bash
# PAIONP (plans/2026-10-13-paionp-prereg.md): RSGPAIonp-L18-s{0,1} = RSG-L18's gentle recipe from RB0-F-s0 on the mixed pool on1_RSGPAIonp
# (on1_L18RB0 rows `self` as RSG + on1_PAIonp = RB0-F-s0's own PAI train160 states, `both`: imitation toward the human-log recovery target),
# PAI rows ~30 % of the draws, turn-approach rows x2. Box, repo root.
#   rf_paionp.sh prep      CPU: rf_paionp.py tag + pack, rf_pait.py merge (pool job)
#   rf_paionp.sh submit    R1, both trainings, held 460 / 18-decision reads, public 1491 / 10-decision guard, P1 open-loop read, report
#   rf_paionp.sh report    the report job (held18 confirm, rf_follow, rf_inside, guard confirm)
set -uo pipefail; cd "$(dirname "$0")/../../.."
RF=$DATA_DIR/runs/route_follow; O=$RF/paionp; G=$RF/guard; M=$DATA_DIR/runs/route_mem/read; E=experiments/onpol_long; W=$DATA_DIR/runs/onpol_long
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; RES=experiments/route_follow/results/paionp
REC="--init RB0-F-s0 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4"
NEW="RSGPAIonp-L18-s0 RSGPAIonp-L18-s1"
HELD="$NEW RSG-L18-s0 RSG-L18-s1 RSGboth-L18-s0 RSGboth-L18-s1 RSGPAIT-L18-s0 RSGPAIT-L18-s1 RB0-F-s0 RB0-F-s1"
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner paionp --name "$name" --tries 1 --priority 14 --no-check \
       --log-dir $O/pool/$name "$@" | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
mkdir -p $O/pool
case $1 in
prep)
  $TR experiments/route_follow/scripts/rf_paionp.py tag || exit 1
  $TR experiments/route_follow/scripts/rf_paionp.py pack || exit 1
  $TR experiments/route_follow/scripts/rf_pait.py merge --pools on1_L18RB0:s:self on1_PAIonp:p:both --out RSGPAIonp || exit 1 ;;
submit)
  ONW=$(python3 -c "import json; print(json.load(open('$DATA_DIR/runs/op_parity/cache/on1_PAIonp/build.json'))['on_weight'])")
  ARM="--on on1_RSGPAIonp --on-k 13 --on-split alpasim_nuplan/onpol-train --on-mode mixed --on-weight $ONW"
  sub pon-r1 --vram 12 --cpu 4 --ram 40 --timeout-h 1 --gpus 1,2,3,4,5 -- $TR experiments/route_follow/scripts/rf_paionp.py r1 --out $RES > /dev/null || exit 1
  dep=(); tr=()
  for T in $NEW; do s=${T##*-s}
    t=$(sub pon-tr-$T --train --vram 28 --cpu 6 --ram 70 --timeout-h 2 --gpus 1,2,3,4,5 -- $TR experiments/route_mem/scripts/rm_train.py train --tag $T $REC $ARM --seed $s) || exit 1
    tr+=($t)
    PRIO=14 bash experiments/route_mem/scripts/rm_chain.sh submit-read $T $t || exit 1
    dep+=($(grep " rm-cover-$T$" $DATA_DIR/runs/route_mem/jobs.txt | tail -1 | cut -d' ' -f1))
    for i in $(seq 0 9); do
      dep+=($(sub pon-g10-$T-$i --after $t --vram 27 --cpu 8 --ram 30 --timeout-h 2 --gpus 1,2,3,4,5 --pin-strict --env CL_HIST_KEY=ol-fast-read -- bash $E/scripts/ol_chain.sh readlist $T \
            $W/lists/all_n10/c$i.txt 10 $G/read10/$T/b$i)) || exit 1
    done
  done
  sub pon-p1ol --after "$(IFS=,; echo "${tr[*]}")" --vram 12 --cpu 4 --ram 40 --timeout-h 1 --gpus 1,2,3,4,5 -- $TR experiments/route_follow/scripts/rf_pai_openloop.py \
      --p1 $RF/p1_openloop/rsg_p1 --logged RSG-L18-s0 --tags RSG-L18-s0 $NEW RSGPAIT-L18-s0 --name p1_paionp --out $RES > /dev/null || exit 1
  sub pon-report --after "$(IFS=,; echo "${dep[*]}")" --vram 0.5 --cpu 16 --ram 32 --timeout-h 1 -- bash experiments/route_follow/scripts/rf_paionp.sh report > /dev/null || exit 1
  cat $O/jobs.txt ;;
report)
  R=""; for t in $HELD; do R="$R $t=$(ls -d $M/$t/c*/run | paste -sd,)"; done
  python3 $RF/held.py $R > $O/held_raw.txt; rc0=$?
  H=(); F=(); for t in $HELD; do d=$(ls -d $M/$t/c*/run | paste -sd,); H+=("$t@held18=$d"); F+=("$t@18=$d"); done
  $P $E/scripts/ol_confirm.py --runs "${H[@]}" --base RB0-F-s0 --mean RSGPAIonp-L18=RSGPAIonp-L18-s0,RSGPAIonp-L18-s1 RSG-L18=RSG-L18-s0,RSG-L18-s1 \
     RSGboth-L18=RSGboth-L18-s0,RSGboth-L18-s1 RSGPAIT-L18=RSGPAIT-L18-s0,RSGPAIT-L18-s1 RB0-F=RB0-F-s0,RB0-F-s1 \
     --pairs RSGPAIonp-L18:RSG-L18 RSGPAIonp-L18-s0:RSG-L18-s0 RSGPAIonp-L18-s1:RSG-L18-s1 RSGPAIonp-L18:RSGPAIT-L18 RSGPAIonp-L18:RSGboth-L18 --out $O/held18; rc1=$?
  $P experiments/route_follow/scripts/rf_follow.py --runs "${F[@]}" --out $O/follow --workers 48 --pairs RSGPAIonp-L18-s0@18:RSG-L18-s0@18 RSGPAIonp-L18-s1@18:RSG-L18-s1@18 \
     RSGPAIonp-L18-s0@18:RB0-F-s0@18 RSGPAIonp-L18-s1@18:RB0-F-s0@18; rc2=$?
  $P experiments/route_follow/scripts/rf_inside.py --follow $O/follow --out $O/inside.json \
     --pairs RSGPAIonp-L18-s0@18:RSG-L18-s0@18 RSGPAIonp-L18-s1@18:RSG-L18-s1@18; rc3=$?
  A=$DATA_DIR/runs/alpasim; Q=()
  Q+=("YR10m10-F-s0@all10=$(python3 -c "import json; a = json.load(open('$A/ot3/a/manifest.json')); b = json.load(open('$A/cf1/a/manifest.json')); print(','.join(a['YR10m10-F-s0'] + a['YR10m10-F-s0@new'] + b['YR10m10-F-s0@fresh']))")")
  for T in RSG-L18-s0 RSG-L18-s1 RSGPAIT-L18-s0 RSGPAIT-L18-s1 RSGboth-L18-s0 RSGboth-L18-s1 $NEW; do Q+=("$T@all10=$(ls -d $G/read10/$T/b*/ | sed 's#/$##' | paste -sd,)"); done
  $P $E/scripts/ol_confirm.py --runs "${Q[@]}" --base YR10m10-F-s0 --mean RSG-L18=RSG-L18-s0,RSG-L18-s1 RSGPAIT-L18=RSGPAIT-L18-s0,RSGPAIT-L18-s1 \
     RSGboth-L18=RSGboth-L18-s0,RSGboth-L18-s1 RSGPAIonp-L18=RSGPAIonp-L18-s0,RSGPAIonp-L18-s1 \
     --pairs RSGPAIonp-L18:RSG-L18 RSGPAIonp-L18-s0:RSG-L18-s0 RSGPAIonp-L18-s1:RSG-L18-s1 RSGPAIonp-L18:RSGPAIT-L18 --out $O/guard; rc4=$?
  echo "held $rc0 held18 $rc1 follow $rc2 inside $rc3 guard $rc4"; exit $((rc0 + rc1 + rc2 + rc3 + rc4)) ;;
*) echo "rf_paionp.sh prep | submit | report" >&2; exit 2 ;;
esac
