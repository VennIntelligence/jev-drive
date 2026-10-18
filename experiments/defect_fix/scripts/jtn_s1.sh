#!/usr/bin/env bash
# JTN-L18-s1: the second seed of JTN-L18-s0 (decision 359). Box, repo root. Calls jt.sh unchanged (no edit to it): the training command is
# jt.sh's own DRY=1 print with `--seed 0` -> `--seed 1`; the reads are jt.sh's aread / rm_chain read / sum bodies, plus the PAI 281 ACC + GAP read.
#   jtn_s1.sh submit [after]   train (prio 10) -> arena held stop3 + off (18 chunks each) + held 460 @18 (6) + PAI 281 ACC+GAP (8 chunks) (prio 11) -> summary
#   jtn_s1.sh train | check | sum   job bodies (check: DRY diff against JTN-L18-s0's command)
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
T=JTN-L18-s1; T0=JTN-L18-s0; O=$DATA_DIR/runs/defect_fix; J=$O/jt; JT=experiments/defect_fix/scripts/jt.sh; ME=experiments/defect_fix/scripts/jtn_s1.sh
AH=$DATA_DIR/runs/alpasim_proxy/arena/hold; R=experiments/defect_fix/results/jtb/$T
TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
LAT=$O/lat/L18NR1v2; PAISERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner defect-fix --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-10} \
     --no-check --log-dir $J/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $J/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
cmd() { DRY=1 ARCROWS=brake bash $JT train $1 on1_JT $LAT JT00 9500 0 | tail -1; }
case ${1:-} in
check)
  diff <(cmd $T0 | tr ' ' '\n') <(cmd $T | sed 's/ --seed 0$/ --seed 1/' | tr ' ' '\n') ; echo "(expected: --tag and --seed lines only)" ;;
train)
  C=$(cmd $T | sed 's/ --seed 0$/ --seed 1/'); [[ $C == *" --seed 1" && $C == *"--tag $T "* ]] || die "bad command: $C"
  echo "$(date '+%F %T') train $T: $C" | tee -a $J/stages.txt
  exec env ARCROWS=brake $C ;;
submit)
  A=${2:-}; [[ ! -e $DATA_DIR/runs/op_parity/runs/$T/ckpt-final.pt ]] || die "$T exists"; mkdir -p $J/pool
  t=$(sub jt-train-$T ${A:+--after $A} --train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train --env ARCROWS=brake --timeout-h 3 -- bash $ME train) || exit 1
  ids=(); nch=$(python3 -c "import json; print(len(json.load(open('$AH/meta.json'))['chunks']))")
  for V in stop3 off; do for c in $(seq 0 $((nch - 1))); do
    ids+=($(PRIO=11 TRIES=2 sub jt-ar-$T-$V-c$c --after $t --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 -- bash $JT aread $T $V $c)) || exit 1; done; done
  for i in 0 1 2 3 4 5; do
    ids+=($(PRIO=11 TRIES=2 sub jt-held-$T-$i --after $t --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
        -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1; done
  mkdir -p $O/pai_accgap; AFTER=$t CHUNKS_FROM=$O/pai_accgap/$T0/chunks SERVE="$PAISERVE" WIZ=controller.gains.rel_front_steering_angle_weight=10 CONC=4 PRIO=11 \
    bash experiments/alpasim/scripts/pai_box.sh submit $O/pai_accgap/$T $T > $J/pai_$T.submit.txt || die "pai submit"
  pp=$(awk '$1 ~ /^[0-9]{4}-|^[0-9a-f]+$/ {print $1}' $J/pai_$T.submit.txt | paste -sd,); [[ -n $pp ]] || die "no PAI job"
  awk '$2 ~ /^pai-/' $J/pai_$T.submit.txt >> $J/jobs.txt
  s=$(PRIO=11 sub jt-sum-$T --after $(csv ${ids[@]}),$pp --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME sum) || exit 1
  echo "$(date '+%F %T') submit $T: train $t reads ${ids[0]} .. ${ids[-1]} (${#ids[@]}) PAI $pp summary $s" | tee -a $J/stages.txt ;;
sum)
  mkdir -p $R   # same bodies as jt.sh sum, with s0 alongside s1 in the arena and held tables
  $AP experiments/defect_fix/scripts/jt.py arena --tags $T@stop3 $T@off $T0@stop3 $T0@off --out $J/report/$T && cp $J/report/$T/arena.md $R/arena.md || die "arena"
  $TR experiments/defect_fix/scripts/jt.py held --tags $T $T0 > $R/held.txt 2>&1 || die "held"; cat $R/held.txt
  .venv/bin/python experiments/defect_fix/scripts/pai281.py --j $O/pai_accgap/$T --jl "$T+ACC+GAP" --j2 $O/pai_accgap/$T0 --j2l "$T0+ACC+GAP" \
    --ref $DATA_DIR/runs/route_follow/acc_gap/read/s2 --rl RSG+ACC+GAP --out $R/pai281_accgap.md || die "pai281"
  { echo; echo "## Seed 0 vs seed 1 (arena.md / held.txt list s1 and s0 side by side, both against RSG-L18-s0)"; echo "pai281 middle column = JTN-L18-s0, reference = RSG+ACC+GAP"; } >> $R/pai281_accgap.md
  ;;
*) sed -n 2,6p $0; exit 2 ;;
esac
