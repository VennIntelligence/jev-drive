#!/usr/bin/env bash
# First round under the new split (plans/2026-10-13-allrows-prereg.md): 4 trainings RSGall-L18 / RSGallV-L18 x seeds 0, 1 on the all-1491 pool,
# each followed (--after) by the public 1491 / 10-decision read (10 chunks, ol_chain.sh readlist, tmp/rn/guard.sh's way), then one confirm job
# (ol_confirm.py vs the stored YR10m10-F-s0 rollouts, RSG-L18 guard reads as the reference) and one deficit job (rf_deficit.py). Box, repo root:
#   bash experiments/route_follow/scripts/rf_allrows.sh submit      -> job ids in $O/jobs.txt
#   bash experiments/route_follow/scripts/rf_allrows.sh confirm | deficit   (job bodies)
set -uo pipefail; cd "$(dirname "$0")/../../.."
E=experiments/onpol_long; W=$DATA_DIR/runs/onpol_long; G=$DATA_DIR/runs/route_follow/guard; O=$DATA_DIR/runs/route_follow/allrows
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python
TAGS="RSGall-L18-s0 RSGall-L18-s1 RSGallV-L18-s0 RSGallV-L18-s1"
REC="--init RB0-F-s0 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on on1_L18RSGall --on-k 13 --on-split alpasim_nuplan/onpol-all --on-mode self"
dirs() { ls -d $1/$2/b*/ | sed 's#/$##' | paste -sd,; }
MEAN=(RSG-L18=RSG-L18-s0,RSG-L18-s1 RSGall-L18=RSGall-L18-s0,RSGall-L18-s1 RSGallV-L18=RSGallV-L18-s0,RSGallV-L18-s1)
PAIRS=(RSGall-L18:RSG-L18 RSGallV-L18:RSG-L18 RSGallV-L18:RSGall-L18)
case $1 in
submit)
  mkdir -p $O; sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner next-round --name "$name" --tries 1 --priority 12 --no-check \
       --log-dir $O/pool/$name "$@" | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
  rr=()
  for T in $TAGS; do
    s=${T##*-s}; X=(); [[ $T == RSGallV-* ]] && X=(--ego-drop-speed 0.5)
    t=$(sub tr-$T --train --vram 28 --cpu 6 --ram 70 --timeout-h 2 -- $TR experiments/route_mem/scripts/rm_train.py train --tag $T $REC "${X[@]}" --seed $s) || exit 1
    for i in $(seq 0 9); do
      rr+=($(sub ar10-$T-$i --after $t --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 -- bash $E/scripts/ol_chain.sh readlist $T $W/lists/all_n10/c$i.txt 10 $O/read10/$T/b$i)) || exit 1
    done
  done
  a=$(IFS=,; echo "${rr[*]}")
  sub ar10-confirm --after "$a" --vram 0.5 --cpu 16 --ram 32 --timeout-h 1 -- bash experiments/route_follow/scripts/rf_allrows.sh confirm > /dev/null || exit 1
  sub ar10-deficit --after "$a" --vram 0.5 --cpu 32 --ram 32 --timeout-h 1 -- bash experiments/route_follow/scripts/rf_allrows.sh deficit > /dev/null || exit 1
  cat $O/jobs.txt ;;
confirm)
  A=$DATA_DIR/runs/alpasim; R=()
  R+=("YR10m10-F-s0@all10=$(python3 -c "import json; a = json.load(open('$A/ot3/a/manifest.json')); b = json.load(open('$A/cf1/a/manifest.json')); print(','.join(a['YR10m10-F-s0'] + a['YR10m10-F-s0@new'] + b['YR10m10-F-s0@fresh']))")")
  for T in RSG-L18-s0 RSG-L18-s1; do R+=("$T@all10=$(dirs $G/read10 $T)"); done
  for T in $TAGS; do R+=("$T@all10=$(dirs $O/read10 $T)"); done
  $P $E/scripts/ol_confirm.py --runs "${R[@]}" --base YR10m10-F-s0 --mean "${MEAN[@]}" --pairs "${PAIRS[@]}" --out $O/confirm ;;
deficit)
  R=(); for T in RSG-L18-s0 RSG-L18-s1; do R+=("$T=$(dirs $G/read10 $T)"); done
  for T in $TAGS; do R+=("$T=$(dirs $O/read10 $T)"); done
  $P experiments/route_follow/scripts/rf_deficit.py --runs "${R[@]}" --mean "${MEAN[@]}" --pairs "${PAIRS[@]}" --out $O/deficit --workers 32 ;;
*) echo "rf_allrows.sh submit | confirm | deficit" >&2; exit 2 ;;
esac
