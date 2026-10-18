#!/usr/bin/env bash
# NTV0-L18-s1: the second seed of NTV0-L18-s0 (decision 366). Box, repo root. Calls jt.sh unchanged: the training command is jt.sh's own DRY=1 print
# for NTV0-L18-s0 (ONK=32, draw JT00S, ARCROWS=brake) with only the tag and `--seed 0` -> `--seed 1` changed; reads = rm_chain.sh read (held 460 @18) + PAI 281 ACC+GAP.
#   ntv0_s1.sh submit [after]   train (prio 12) -> held 460 @18 (6 chunks, prio 13) -> PAI stage (prio 13) -> summary; no arena reads
#   ntv0_s1.sh train | check | pai | sum   job bodies (check: DRY diff against NTV0-L18-s0's command)
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
T=NTV0-L18-s1; T0=NTV0-L18-s0; TJ=JTN-L18-s0; O=$DATA_DIR/runs/defect_fix; J=$O/jt; JT=experiments/defect_fix/scripts/jt.sh; ME=experiments/defect_fix/scripts/ntv0_s1.sh
R=experiments/defect_fix/results/ntv/$T; TR=$DATA_DIR/envs/op-train/bin/python
LAT=$O/lat/L18NR1v2; PAISERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner defect-fix --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-12} \
     --no-check --log-dir $J/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $J/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
cmd() { DRY=1 ONK=32 ARCROWS=brake bash $JT train $1 on1_JT $LAT JT00S 9500 0 | tail -1; }
case ${1:-} in
check)
  diff <(cmd $T0 | tr ' ' '\n') <(cmd $T | sed 's/ --seed 0$/ --seed 1/' | tr ' ' '\n') ; echo "(expected: --tag and --seed lines only)" ;;
train)
  C=$(cmd $T | sed 's/ --seed 0$/ --seed 1/'); [[ $C == *" --seed 1" && $C == *"--tag $T "* ]] || die "bad command: $C"
  echo "$(date '+%F %T') train $T: $C" | tee -a $J/stages.txt
  exec env ARCROWS=brake ONK=32 $C ;;
submit)
  A=${2:-}; [[ ! -e $DATA_DIR/runs/op_parity/runs/$T/ckpt-final.pt ]] || die "$T exists"; mkdir -p $J/pool
  [[ -e $O/w/JT00S.npz ]] || die "missing weights JT00S.npz"
  t=$(PRIO=12 sub ntv0s1-train ${A:+--after $A} --train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train --env ARCROWS=brake --env ONK=32 --timeout-h 3 -- bash $ME train) || exit 1
  ids=()
  for i in 0 1 2 3 4 5; do
    ids+=($(PRIO=13 TRIES=2 sub ntv0s1-held-$i --after $t --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
        -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1; done
  echo "${ids[*]}" > $J/ntv0s1_reads.txt     # pai_box.sh submit needs the checkpoint, so the PAI reads and the summary are submitted by a stage job after training
  pj=$(PRIO=13 sub ntv0s1-pai --after $t --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME pai) || exit 1
  echo "$(date '+%F %T') submit $T: train $t held ${ids[0]} .. ${ids[-1]} PAI stage $pj" | tee -a $J/stages.txt ;;
pai)
  mkdir -p $O/pai_accgap; CHUNKS_FROM=$O/pai_accgap/$T0/chunks SERVE="$PAISERVE" WIZ=controller.gains.rel_front_steering_angle_weight=10 CONC=4 PRIO=13 \
    bash experiments/alpasim/scripts/pai_box.sh submit $O/pai_accgap/$T $T > $J/pai_$T.submit.txt || die "pai submit"
  pp=$(awk '$2 ~ /^pai-/ {print $1}' $J/pai_$T.submit.txt | paste -sd,); [[ -n $pp ]] || die "no PAI job"
  awk '$2 ~ /^pai-/' $J/pai_$T.submit.txt >> $J/jobs.txt
  s=$(PRIO=13 sub ntv0s1-sum --after $(csv $(cat $J/ntv0s1_reads.txt)),$pp --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME sum) || exit 1
  echo "$(date '+%F %T') pai $T: PAI $pp summary $s" | tee -a $J/stages.txt ;;
sum)
  mkdir -p $R
  $TR experiments/defect_fix/scripts/jt.py held --tags $T $T0 > $R/held.txt 2>&1 || die "held"; cat $R/held.txt
  P=experiments/defect_fix/scripts/pai281.py; REF=$DATA_DIR/runs/route_follow/acc_gap/read/s2
  .venv/bin/python $P --j $O/pai_accgap/$T --jl "$T+ACC+GAP" --j2 $O/pai_accgap/$T0 --j2l "$T0+ACC+GAP" --ref $REF --rl RSG+ACC+GAP --out $R/pai281_accgap.md || die "pai281 vs s0"
  .venv/bin/python $P --j $O/pai_accgap/$T --jl "$T+ACC+GAP" --j2 $O/pai_accgap/$TJ --j2l "$TJ+ACC+GAP" --ref $REF --rl RSG+ACC+GAP --out $R/pai281_accgap_vs_jtn.md || die "pai281 vs jtn"
  date '+%F %T' > $R/DONE; ls -l $R ;;
*) sed -n 2,6p $0; exit 2 ;;
esac
