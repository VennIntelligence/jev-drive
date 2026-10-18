#!/usr/bin/env bash
# NTV0-L18-s0 exploratory winner reads (NR1 prereg supplement 14, addendum 1). Box, repo root. Calls jt.sh unchanged (aread / sum bodies) and
# pai_box.sh with jtn_s1.sh's probe serve config; checkpoint already trained, so everything is submitted at once.
#   ntv0_reads.sh submit   arena held stop3 + off (18 chunks each) + PAI 281 ACC+GAP (8 chunks), prio 13 -> summary job (prio 13)
#   ntv0_reads.sh sum      job body: results/ntv/NTV0-L18-s0/{arena.md,held.txt,pai281_accgap.md}
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
T=NTV0-L18-s0; T0=JTN-L18-s0; O=$DATA_DIR/runs/defect_fix; J=$O/jt; JT=experiments/defect_fix/scripts/jt.sh; ME=experiments/defect_fix/scripts/ntv0_reads.sh
AH=$DATA_DIR/runs/alpasim_proxy/arena/hold; R=experiments/defect_fix/results/ntv/$T; PRIO=${PRIO:-13}
PAISERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner defect-fix --name "$name" --tries ${TRIES:-1} --priority $PRIO \
     --no-check --log-dir $J/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $J/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
case ${1:-} in
submit)
  [[ -e $DATA_DIR/runs/op_parity/runs/$T/ckpt-final.pt ]] || die "$T has no ckpt-final.pt"; mkdir -p $J/pool $O/pai_accgap; ids=()
  nch=$(python3 -c "import json; print(len(json.load(open('$AH/meta.json'))['chunks']))")
  for V in stop3 off; do for c in $(seq 0 $((nch - 1))); do
    ids+=($(TRIES=2 sub ntv0-ar-$V-c$c --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 -- bash $JT aread $T $V $c)) || exit 1; done; done
  CHUNKS_FROM=$O/pai_accgap/$T0/chunks SERVE="$PAISERVE" WIZ=controller.gains.rel_front_steering_angle_weight=10 CONC=4 PRIO=$PRIO \
    bash experiments/alpasim/scripts/pai_box.sh submit $O/pai_accgap/$T $T > $J/pai_$T.submit.txt || die "pai submit"
  pp=$(awk '$2 ~ /^pai-/ {print $1}' $J/pai_$T.submit.txt | paste -sd,); [[ -n $pp ]] || die "no PAI job"
  awk '$2 ~ /^pai-/' $J/pai_$T.submit.txt >> $J/jobs.txt
  s=$(sub ntv0-sum --after $(csv ${ids[@]}),$pp --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME sum) || exit 1
  echo "$(date '+%F %T') ntv0 reads: arena ${ids[0]} .. ${ids[-1]} (${#ids[@]}) PAI $pp summary $s" | tee -a $J/stages.txt ;;
sum)
  mkdir -p $R
  bash $JT sum $T "stop3 off" ntv || die "arena summary"
  .venv/bin/python experiments/defect_fix/scripts/pai281.py --j $O/pai_accgap/$T --jl "$T+ACC+GAP" --j2 $O/pai_accgap/$T0 --j2l "$T0+ACC+GAP" \
    --ref $DATA_DIR/runs/route_follow/acc_gap/read/s2 --rl RSG+ACC+GAP --out $R/pai281_accgap.md || die "pai281"
  date '+%F %T' > $R/DONE; ls -l $R ;;
*) sed -n 2,6p $0; exit 2 ;;
esac
