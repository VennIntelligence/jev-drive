#!/usr/bin/env bash
# DESIRE-D0 chain (plans/2026-10-17-desire-d0-prereg.md). Box, repo root; every GPU job through the pool, --after chained; ids in $O/jobs.txt.
#   dd_chain.sh low           low-speed open loop: lowrows (CPU) -> lowplans (GPU) -> lowreport (CPU) -> $O/LOW_DONE (lowspeed.md / json in $O/low)
#   dd_chain.sh gate          G-none / G-feed (GPU, dd.py gates) -> $O/gates.json
#   dd_chain.sh d0 <v_min>    after the gate job: PAI 281 D0 (acc's chunk lists) + c000 re-read without desire, train160 D0 + ref, held D0 + ref,
#                             then the report job -> $O/D0_DONE
# Env: PRIO (13), AFTER (pool ids the d0 jobs wait for).
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}
O=$DATA_DIR/runs/desire_cap/d0; S=experiments/desire_cap/scripts; V=.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; T=RSG-L18-s0
ACC="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1"; WIZ=controller.gains.rel_front_steering_angle_weight=10; PRIO=${PRIO:-13}
mkdir -p $O/pool
sub() { local name=$1 after=$2; shift 2; local id; id=$($V -m jevdrive.cl submit --owner desire-d0 --name "$name" --tries 1 --priority $PRIO --no-check \
        --log-dir $O/pool/$name ${after:+--after $after} "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }
        echo "$id $name" >> $O/jobs.txt; echo "$id"; }
ids() { awk '{print $1}' "$1/jobs.txt" | paste -sd, -; }
case ${1:-} in
low)
  i1=$(sub dd-lowrows "" --vram 0.5 --cpu 2 --ram 16 --timeout-h 1 -- $V $S/dd.py lowrows) || exit 1
  i2=$(sub dd-lowplans "$i1" --vram 16 --cpu 4 --ram 48 --timeout-h 2 -- $TR $S/dd.py lowplans) || exit 1
  i3=$(sub dd-lowreport "$i2" --vram 0.5 --cpu 2 --ram 16 --timeout-h 1 -- bash -c "$V $S/dd.py lowreport --out $O/low && touch $O/LOW_DONE || touch $O/LOW_ERROR") || exit 1
  echo "low: $i1 $i2 $i3" ;;
gate)
  sub dd-gates "" --vram 16 --cpu 4 --ram 32 --timeout-h 1 -- $TR $S/dd.py gates --out $O/gates.json ;;
d0)
  VMIN=${2:?v_min}; DS="JEV_DESIRE=1 JEV_DESIRE_VMIN=$VMIN"; R=$DATA_DIR/runs/route_follow/pai_g10/acc; PB=experiments/alpasim/scripts/pai_box.sh
  [[ ! -e $O/pai281 ]] || { echo "$O/pai281 exists" >&2; exit 1; }
  CHUNKS_FROM=$R/chunks SERVE="$ACC $DS" WIZ=$WIZ PRIO=$PRIO AFTER=${AFTER:-} bash $PB submit $O/pai281 $T || exit 1
  mkdir -p $O/acc_re/chunks && cp $R/chunks/c000.txt $O/acc_re/chunks/
  CHUNKS_FROM=$O/acc_re/chunks SERVE="$ACC" WIZ=$WIZ PRIO=$PRIO AFTER=${AFTER:-} bash $PB submit $O/acc_re $T || exit 1
  CHUNK=23 SERVE="$ACC $DS" WIZ=$WIZ PRIO=$PRIO AFTER=${AFTER:-} bash $PB submit $O/pai160 $T $O/lists/pai160.txt || exit 1
  CHUNKS_FROM=$O/pai160/chunks SERVE="$ACC" WIZ=$WIZ PRIO=$PRIO AFTER=${AFTER:-} bash $PB submit $O/pai160_ref $T || exit 1
  RM=experiments/route_mem/scripts/rm_chain.sh; h=()
  split -n l/2 -d --additional-suffix=.txt $O/lists/held.txt $O/lists/held_c
  for x in ref d0; do for c in 00 01; do
    ev=(--env CL_HIST_KEY=alpasim-read4); [[ $x == d0 ]] && ev+=(--env JEV_DESIRE=1 --env JEV_DESIRE_VMIN=$VMIN)
    h+=($(sub dd-held-$x-$c "${AFTER:-}" --vram 20 --cpu 6 --ram 40 --timeout-h 2 "${ev[@]}" -- bash $RM readlist $T $O/lists/held_c$c.txt $O/held/$x/c$c)) || exit 1
  done; done
  all="$(ids $O/pai281),$(ids $O/acc_re),$(ids $O/pai160),$(ids $O/pai160_ref),$(IFS=,; echo "${h[*]}")"
  r=$(sub dd-d0report "$all" --vram 0.5 --cpu 16 --ram 48 --timeout-h 3 -- bash $S/dd_chain.sh report $VMIN) || exit 1
  printf '%s\n' "code=$(git rev-parse --short HEAD)" "v_min=$VMIN" "serve=$ACC $DS" "wiz=$WIZ" > $O/config.txt
  echo "d0 report $r" ;;
report)   # job body: rf_tax extracts of every arm (AlpaSim venv), lanes of the train160 scenes, then dd.py d0report
  VMIN=${2:?v_min}; P=$DATA_DIR/third_party/alpasim/.venv/bin/python; RF=experiments/route_follow/scripts; X=$O/extract; mkdir -p $X
  ln -sfn $DATA_DIR/runs/route_follow/acc_gap/acc.pkl $X/acc.pkl
  ex() { [[ -s $X/$1.pkl ]] || $P $RF/rf_tax.py extract --runs "${@:2}" --out $X/$1.pkl --workers 16 > $X/$1.log 2>&1; }
  ex pai281_d0 $O/pai281/runs/*/sim; ex pai160_d0 $O/pai160/runs/*/sim; ex pai160_ref $O/pai160_ref/runs/*/sim
  ex held_d0 $O/held/d0/c*; ex held_ref $O/held/ref/c*
  [[ -s $X/lanes_pai160.pkl ]] || $P $RF/rf_laneoff.py lanes --runs $O/pai160_ref/runs/*/sim --out $X/lanes_pai160.pkl --workers 8 > $X/lanes.log 2>&1
  $V $S/dd.py d0report --vmin $VMIN --reidentity --out $O/report && touch $O/D0_DONE || { touch $O/D0_ERROR; exit 1; } ;;
*) sed -n '2,8p' "$0"; exit 2 ;;
esac
