#!/usr/bin/env bash
# RSGboth variants (plans/2026-10-13-rsgboth-prereg.md Amendment 1): RSGboth-L18-s0's exact command plus ONE change each, seed 0:
#   RSGbA30-L18-s0  --agent-lam 30                                              (stronger agent hinge)
#   RSGbW-L18-s0    --on on1_L18RB0t --on-weight rclass=straight:0.3,rclass=curve:0.3   (non-turn pool rows drawn at 0.3x; same rows, same targets)
#   RSGbC2-L18-s0   --on on1_L18RB0c2                                           (non-turn rows' imitation target capped at the row's own speed)
# Pools: rf_rsgbv.py pool (gate R1 in $DATA_DIR/runs/op_parity/cache/<pool>.r1.json). Each train job (60-step preflight smoke) is followed (--after)
# by the held 460 / 18-decision read (rm_chain.sh submit-read) and the public 1491 / 10-decision guard (10 ol_chain readlist chunks); one report
# job at the end: held.py, ol_confirm held18 + guard, rf_follow, rf_inside, zeros_mech collect + rf_rsgbv.py diag. Box, repo root.
#   rf_rsgbv.sh submit | report
set -uo pipefail; cd "$(dirname "$0")/../../.."
RF=$DATA_DIR/runs/route_follow; O=$RF/rsgboth_v; G=$RF/guard; M=$DATA_DIR/runs/route_mem/read; E=experiments/onpol_long; W=$DATA_DIR/runs/onpol_long
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python
BASE="--init RB0-F-s0 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --shape --ho-w 3 --rm-lr 3e-4 --on-k 13 --on-split alpasim_nuplan/onpol-train --on-mode both --agent-on-only --seed 0"
declare -A X=([RSGbA30-L18-s0]="--agent-lam 30 --on on1_L18RB0" [RSGbW-L18-s0]="--agent-lam 10 --on on1_L18RB0t --on-weight rclass=straight:0.3,rclass=curve:0.3"
              [RSGbC2-L18-s0]="--agent-lam 10 --on on1_L18RB0c2")
NEW="RSGbA30-L18-s0 RSGbW-L18-s0 RSGbC2-L18-s0"
REF="RSGboth-L18-s0 RSG-L18-s0"
case $1 in
submit)
  mkdir -p $O/pool; sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner route-follow --name "$name" --tries 1 --priority 14 --no-check \
       --log-dir $O/pool/$name "$@" | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
  dep=()
  for T in $NEW; do
    t=$(sub rsgbv-$T --train --vram 28 --cpu 6 --ram 70 --timeout-h 3 \
        --preflight "$TR experiments/route_mem/scripts/rm_train.py train --tag ${T%-s0}-smoke $BASE ${X[$T]} --steps 60 --eval-every 60" \
        -- $TR experiments/route_mem/scripts/rm_train.py train --tag $T $BASE ${X[$T]}) || exit 1
    PRIO=14 bash experiments/route_mem/scripts/rm_chain.sh submit-read $T $t || exit 1
    dep+=($(grep " rm-cover-$T$" $DATA_DIR/runs/route_mem/jobs.txt | tail -1 | cut -d' ' -f1))
    for i in $(seq 0 9); do
      dep+=($(sub rsgbv-g10-$T-$i --after $t --vram 27 --cpu 8 --ram 30 --timeout-h 2 --gpus 1,2,3,4,5 --pin-strict --env CL_HIST_KEY=ol-fast-read -- bash $E/scripts/ol_chain.sh readlist $T \
            $W/lists/all_n10/c$i.txt 10 $G/read10/$T/b$i)) || exit 1
    done
  done
  sub rsgbv-report --after "$(IFS=,; echo "${dep[*]}")" --vram 0.5 --cpu 16 --ram 48 --timeout-h 2 -- bash experiments/route_follow/scripts/rf_rsgbv.sh report > /dev/null || exit 1
  cat $O/jobs.txt ;;
report)
  ALL="$NEW $REF"
  R=""; for t in $ALL; do R="$R $t=$(ls -d $M/$t/c*/run | paste -sd,)"; done
  python3 $RF/held.py $R > $O/held_raw.txt; rc0=$?
  H=(); F=(); PH=(); PF=(); for t in $ALL; do d=$(ls -d $M/$t/c*/run | paste -sd,); H+=("$t@held18=$d"); F+=("$t@18=$d"); done
  for t in $NEW; do for r in $REF; do PH+=("$t:$r"); PF+=("$t@18:$r@18"); done; done
  $P $E/scripts/ol_confirm.py --runs "${H[@]}" --base RSG-L18-s0 --pairs "${PH[@]}" --out $O/held18; rc1=$?
  $P experiments/route_follow/scripts/rf_follow.py --runs "${F[@]}" --out $O/follow --workers 16 --pairs "${PF[@]}"; rc2=$?
  $P experiments/route_follow/scripts/rf_inside.py --follow $O/follow --pairs "${PF[@]}" --out $O/inside.json > /dev/null; rc3=$?
  A=$DATA_DIR/runs/alpasim; Q=()
  Q+=("YR10m10-F-s0@all10=$(python3 -c "import json; a = json.load(open('$A/ot3/a/manifest.json')); b = json.load(open('$A/cf1/a/manifest.json')); print(','.join(a['YR10m10-F-s0'] + a['YR10m10-F-s0@new'] + b['YR10m10-F-s0@fresh']))")")
  for T in $ALL; do Q+=("$T@all10=$(ls -d $G/read10/$T/b*/ | sed 's#/$##' | paste -sd,)"); done
  $P $E/scripts/ol_confirm.py --runs "${Q[@]}" --base YR10m10-F-s0 --pairs "${PH[@]}" --out $O/guard; rc4=$?
  C=""; for t in $ALL; do for i in 0 1 2 3 4 5; do C="$C $t~c$i=$M/$t/c$i/run"; done; done
  $P experiments/zeros_mech/scripts/zm_read.py collect --workers 16 --runs $C > $O/zm_collect.log 2>&1; rc5=$?
  Z=$(grep -o "runs/zeros_mech/collect/[0-9-]*" $O/zm_collect.log | tail -1); Z=${Z:+$DATA_DIR/$Z}; [[ -n $Z ]] || Z=$(ls -td $DATA_DIR/runs/zeros_mech/collect/*/ | head -1)
  DP=(); for t in $NEW RSGboth-L18-s0; do DP+=("$t:RSG-L18-s0"); done
  $P experiments/route_follow/scripts/rf_rsgbv.py diag --collect $Z --follow $O/follow --out $O/diag_v --pairs "${DP[@]}" > /dev/null; rc6=$?
  echo "held $rc0 held18 $rc1 follow $rc2 inside $rc3 guard $rc4 zm $rc5 diag $rc6 ($Z)"; exit $((rc0 + rc1 + rc2 + rc3 + rc4 + rc5 + rc6)) ;;
*) echo "rf_rsgbv.sh submit | report" >&2; exit 2 ;;
esac
