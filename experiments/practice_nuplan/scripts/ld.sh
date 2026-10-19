#!/usr/bin/env bash
# LATERAL-DIAG confirmatory run (plans/2026-10-19-pn-prereg.md supplement 2): three 1 000-step continuations of NTV0-L18-s0 that differ only in
# the pool / draw (LDC: on1_JT + JT00S; LDX: on1_JTX + JTX00S; LDXS: on1_JTX + JTX00SNS, the new static rows at weight 0), then one open-loop
# replay on the native-gap dumps and the registered readout. Box, repo root; every job through the pool (priority 11), --after chained.
#   ld.sh check            the three commands (jt.sh train's DRY print, --init RB0-F-s0 -> NTV0-L18-s0) and their pairwise diffs
#   ld.sh submit           wns (CPU) -> 3 trainings -> replay -> confirm; ids in $W/jobs.txt, stages in $W/stages.txt
#   ld.sh train <arm>      job body
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
W=$DATA_DIR/runs/practice_nuplan/lat_diag; ME=experiments/practice_nuplan/scripts/ld.sh; PY=experiments/practice_nuplan/scripts/lat_diag.py
JT=experiments/defect_fix/scripts/jt.sh; TR=$DATA_DIR/envs/op-train/bin/python; LAT=$DATA_DIR/runs/defect_fix/lat/L18NR1v2
declare -A POOL=([LDC]=on1_JT [LDX]=on1_JTX [LDXS]=on1_JTX) DRAW=([LDC]=JT00S [LDX]=JTX00S [LDXS]=JTX00SNS)
mkdir -p $W/pool
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $W/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner practice-nuplan --name "$name" --tries 1 --priority 11 \
     --no-check --log-dir $W/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $W/jobs.txt; echo "$id"; }
cmd() { DRY=1 ONK=32 ARCROWS=brake WARM=50 bash $JT train $1-L18-s0 ${POOL[$1]} $LAT ${DRAW[$1]} 1000 0 | tail -1 | sed 's/--init RB0-F-s0 /--init NTV0-L18-s0 /'; }
case ${1:-} in
check)
  for A in LDC LDX LDXS; do cmd $A > $W/cmd_$A.txt; [[ $(cat $W/cmd_$A.txt) == *"--init NTV0-L18-s0 "*"--steps 1000 "* ]] || die "$A: $(cat $W/cmd_$A.txt)"; done
  for B in LDX LDXS; do
    diff <(tr ' ' '\n' < $W/cmd_LDC.txt) <(tr ' ' '\n' < $W/cmd_$B.txt) > $W/cmd_diff_$B.txt
    n=$(grep -c '^[<>]' $W/cmd_diff_$B.txt); [[ $n -le 6 ]] || die "LDC vs $B differ beyond --tag / --on / --on-wfile"
    grep '^[<>]' $W/cmd_diff_$B.txt | grep -vqE 'L18-s0$|on1_JTX?$|w/JT[A-Z0-9]*\.npz$' && die "unexpected diff tokens LDC vs $B"; done
  cat $W/cmd_diff_LDX.txt $W/cmd_diff_LDXS.txt; echo "diff ok" ;;
submit)
  for A in LDC LDX LDXS; do [[ ! -e $DATA_DIR/runs/op_parity/runs/$A-L18-s0/ckpt-final.pt ]] || die "$A-L18-s0 exists"; done
  w=$(sub ld-wns --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- $TR $PY wns) || exit 1
  t=(); for A in LDC LDX LDXS; do
    t+=($(sub ld-train-$A --after $w --train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train --timeout-h 2 -- bash $ME train $A)) || exit 1; done
  r=$(sub ld-replay --after $(IFS=,; echo "${t[*]}") --vram 10 --cpu 4 --ram 24 --timeout-h 3 -- $TR $PY replay --tags LDC-L18-s0 LDX-L18-s0 LDXS-L18-s0 --out replay_ld.parquet) || exit 1
  c=$(sub ld-confirm --after $r --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- $TR $PY confirm) || exit 1
  echo "$(date '+%F %T') submit: wns $w train ${t[*]} replay $r confirm $c" | tee -a $W/stages.txt ;;
train)
  bash $ME check > /dev/null || exit 1
  C=$(cmd $2); echo "$(date '+%F %T') train $2: $C" | tee -a $W/stages.txt
  exec $C ;;
*) sed -n 2,7p $0; exit 2 ;;
esac
