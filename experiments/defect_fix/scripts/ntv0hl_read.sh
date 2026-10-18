#!/usr/bin/env bash
# NTV0-L18-s0 + ACC + GAP + holdlead on PAI 281 (prereg experiments/defect_fix/plans/2026-10-19-ntv0-holdlead-prereg.md). Box, repo root.
#   ntv0hl_read.sh submit   8 PAI chunks (the stored NTV0 read's lists), prio 12 -> summary job after them
#   ntv0hl_read.sh sum      job body: results/ntv/NTV0-L18-s0-hl/pai281.md, paired vs stored NTV0 + ACC + GAP and RSG + ACC + GAP
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
T=NTV0-L18-s0; O=$DATA_DIR/runs/defect_fix; J=$O/jt; P=$O/pai_accgap; ME=experiments/defect_fix/scripts/ntv0hl_read.sh
R=experiments/defect_fix/results/ntv/$T-hl; PRIO=${PRIO:-12}
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
case ${1:-} in
submit)
  mkdir -p $J/pool
  CHUNKS_FROM=$P/$T/chunks SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7 JEV_VHIST=holdlead" \
    WIZ=controller.gains.rel_front_steering_angle_weight=10 CONC=4 PRIO=$PRIO \
    bash experiments/alpasim/scripts/pai_box.sh submit $P/$T-hl $T > $J/pai_$T-hl.submit.txt || die "pai submit"
  pp=$(awk '$2 ~ /^pai-/ {print $1}' $J/pai_$T-hl.submit.txt | paste -sd,); [[ -n $pp ]] || die "no PAI job"
  awk '$2 ~ /^pai-/' $J/pai_$T-hl.submit.txt >> $J/jobs.txt
  s=$(.venv/bin/python -m jevdrive.cl submit --owner defect-fix --name ntv0hl-sum --tries 1 --priority $PRIO --no-check --log-dir $J/pool/ntv0hl-sum \
      --after $pp --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME sum < /dev/null | awk '{print $NF}') && [[ -n $s ]] || die "summary submit"
  echo "$s ntv0hl-sum" >> $J/jobs.txt
  echo "$(date '+%F %T') ntv0hl: PAI $pp summary $s" | tee -a $J/stages.txt ;;
sum)
  mkdir -p $R
  .venv/bin/python experiments/defect_fix/scripts/pai281.py --j $P/$T-hl --jl "$T+ACC+GAP+HL" --j2 $P/$T --j2l "$T+ACC+GAP" \
    --ref $DATA_DIR/runs/route_follow/acc_gap/read/s2 --rl RSG+ACC+GAP --out $R/pai281.md || die "pai281"
  date '+%F %T' > $R/DONE; ls -l $R ;;
*) sed -n 2,4p $0; exit 2 ;;
esac
