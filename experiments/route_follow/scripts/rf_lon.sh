#!/usr/bin/env bash
# LON (plans/2026-10-13-lon-prereg.md): RSGboth-L18-s{k}'s exact command plus the three timing-only terms of onpol_nuplan lib/on_lonsafe.py
# (--lon-lead / --lon-curve / --lon-stop, weights below = the prereg's). Train job (60-step preflight smoke) -> held 460 / 18-decision read
# (rm_chain.sh submit-read) + public 1491 / 10-decision guard (10 ol_chain readlist chunks) -> one report job: held.py, ol_confirm held18 + guard,
# rf_follow, rf_inside, zeros_mech collect + rf_rsgbv.py diag vs RSG-L18-s{k}. Box, repo root. Every GPU job on cards 1-5.
#   rf_lon.sh submit <seed> | reads <seed> [after id] | report <seed>
set -uo pipefail; cd "$(dirname "$0")/../../.."
K=${2:?seed}; T=RSGbothLon-L18-s$K
RF=$DATA_DIR/runs/route_follow; O=$RF/lon/s$K; G=$RF/guard; M=$DATA_DIR/runs/route_mem/read; E=experiments/onpol_long; W=$DATA_DIR/runs/onpol_long
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python
LON="--lon-lead 0.05 --lon-curve 10 --lon-stop 0.025"                 # prereg "Weights" (rf_lonprobe.py: w at 1x the imitation push)
CMD="--init RB0-F-s0 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-k 13 --on-split alpasim_nuplan/onpol-train --on-mode both --on on1_L18RB0 --seed $K $LON"
REF="RSGboth-L18-s$K RSG-L18-s$K"; [[ $K == 0 ]] && REF="$REF RSGbC2-L18-s0"
case $1 in
submit)
  mkdir -p $O/pool; sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner route-follow --name "$name" --tries 1 --priority 14 --no-check \
       --log-dir $O/pool/$name "$@" | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
  t=$(sub lon-$T --train --vram 28 --cpu 6 --ram 70 --timeout-h 3 --gpus 1,2,3,4,5 \
      --preflight "$TR experiments/route_mem/scripts/rm_train.py train --tag ${T%-s$K}-smoke-s$K $CMD --steps 60 --eval-every 60" \
      -- $TR experiments/route_mem/scripts/rm_train.py train --tag $T $CMD) || exit 1
  bash $0 reads $K $t || exit 1 ;;
reads)  # reads <seed> [after]: the held read, the guard and the report (also alone, e.g. after a cancelled read chain)
  mkdir -p $O/pool; t=${3:-}
  sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner route-follow --name "$name" --tries 1 --priority 14 --no-check \
       --log-dir $O/pool/$name "$@" | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
  PRIO=14 bash experiments/route_mem/scripts/rm_chain.sh submit-read $T $t || exit 1
  dep=($(grep " rm-cover-$T$" $DATA_DIR/runs/route_mem/jobs.txt | tail -1 | cut -d' ' -f1))
  for i in $(seq 0 9); do
    dep+=($(sub lon-g10-$T-$i ${t:+--after $t} --vram 27 --cpu 8 --ram 30 --timeout-h 2 --gpus 1,2,3,4,5 --pin-strict --env CL_HIST_KEY=ol-fast-read -- bash $E/scripts/ol_chain.sh readlist $T \
          $W/lists/all_n10/c$i.txt 10 $G/read10/$T/b$i)) || exit 1
  done
  sub lon-report-$T --after "$(IFS=,; echo "${dep[*]}")" --vram 0.5 --cpu 16 --ram 48 --timeout-h 2 -- bash experiments/route_follow/scripts/rf_lon.sh report $K > /dev/null || exit 1
  cat $O/jobs.txt ;;
report)
  ALL="$T $REF"
  R=""; for t in $ALL; do R="$R $t=$(ls -d $M/$t/c*/run | paste -sd,)"; done
  python3 $RF/held.py $R > $O/held_raw.txt; rc0=$?
  H=(); F=(); PH=(); PF=(); for t in $ALL; do d=$(ls -d $M/$t/c*/run | paste -sd,); H+=("$t@held18=$d"); F+=("$t@18=$d"); done
  for r in $REF; do PH+=("$T:$r"); PF+=("$T@18:$r@18"); done
  $P $E/scripts/ol_confirm.py --runs "${H[@]}" --base RSG-L18-s$K --pairs "${PH[@]}" --out $O/held18; rc1=$?
  $P experiments/route_follow/scripts/rf_follow.py --runs "${F[@]}" --out $O/follow --workers 16 --pairs "${PF[@]}"; rc2=$?
  $P experiments/route_follow/scripts/rf_inside.py --follow $O/follow --pairs "${PF[@]}" --out $O/inside.json > /dev/null; rc3=$?
  A=$DATA_DIR/runs/alpasim; Q=()
  Q+=("YR10m10-F-s0@all10=$(python3 -c "import json; a = json.load(open('$A/ot3/a/manifest.json')); b = json.load(open('$A/cf1/a/manifest.json')); print(','.join(a['YR10m10-F-s0'] + a['YR10m10-F-s0@new'] + b['YR10m10-F-s0@fresh']))")")
  for t in $ALL; do Q+=("$t@all10=$(ls -d $G/read10/$t/b*/ | sed 's#/$##' | paste -sd,)"); done
  $P $E/scripts/ol_confirm.py --runs "${Q[@]}" --base YR10m10-F-s0 --pairs "${PH[@]}" --out $O/guard; rc4=$?
  C=""; for t in $ALL; do for i in 0 1 2 3 4 5; do C="$C $t~c$i=$M/$t/c$i/run"; done; done
  $P experiments/zeros_mech/scripts/zm_read.py collect --workers 16 --runs $C > $O/zm_collect.log 2>&1; rc5=$?
  Z=$(grep -o "runs/zeros_mech/collect/[0-9-]*" $O/zm_collect.log | tail -1); Z=${Z:+$DATA_DIR/$Z}; [[ -n $Z ]] || Z=$(ls -td $DATA_DIR/runs/zeros_mech/collect/*/ | head -1)
  DP=(); for t in $T $REF; do [[ $t == RSG-L18-s$K ]] || DP+=("$t:RSG-L18-s$K"); done
  $P experiments/route_follow/scripts/rf_rsgbv.py diag --collect $Z --follow $O/follow --out $O/diag --pairs "${DP[@]}" > /dev/null; rc6=$?
  echo "held $rc0 held18 $rc1 follow $rc2 inside $rc3 guard $rc4 zm $rc5 diag $rc6 ($Z)"; exit $((rc0 + rc1 + rc2 + rc3 + rc4 + rc5 + rc6)) ;;
*) echo "rf_lon.sh submit <seed> | report <seed>" >&2; exit 2 ;;
esac
