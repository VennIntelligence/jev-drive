#!/usr/bin/env bash
# BEST-OF-N control-continuation replication (plans/2026-10-19-bn-prereg.md, supplement 1). Box, repo root; pool jobs only, priority 12, --after chained.
#   bnc.sh submit           check commands -> train BNC2K-L18-s0 and BNC4K-L18-s1 -> arena held native (18) + held 460 @18 (6) each -> PAI 281 read of BNC2K-L18-s1 -> PAI summary -> report job; ids in $W/cont_jobs.txt, sentinel $W/CONT_DONE (failure: $W/ERROR via the job's own die)
#   bnc.sh check | train <tag> | pai | report | wait   job bodies
# Every arm is bn.sh's BNC2K command (traincmd: jt.sh JTN command on1_JT / JT00, warm-up 50, --seed 1) with only tag, init and steps changed.
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
E=experiments/best_of_n; ME=$E/scripts/bnc.sh; W=$DATA_DIR/runs/best_of_n; O=$DATA_DIR/runs/defect_fix; J=$O/jt; JT=experiments/defect_fix/scripts/jt.sh
CK=$DATA_DIR/runs/op_parity/runs; TR=$DATA_DIR/envs/op-train/bin/python; LAT=$O/lat/L18NR1v2; AH=$O/../alpasim_proxy/arena/hold
PAISERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; R=$E/results; JOBS=$W/cont_jobs.txt
ARMS="BNC2K-L18-s0:JTN-L18-s0:2000 BNC4K-L18-s1:JTN-L18-s1:4000"; REF=BNC2K-L18-s1
RD=(--vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4)
FT=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train --timeout-h 1)
die() { echo "$(date '+%F %T') ERROR $*" >&2; echo "$(date '+%F %T') $*" >> $W/ERROR; exit 1; }
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner best-of-n --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-12} \
     --no-check --log-dir $W/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $JOBS; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
cmd() {  # cmd <tag> <init> <steps>: bn.sh traincmd, verbatim
  DRY=1 ARCROWS=brake WARM=50 bash $JT train $1 on1_JT $LAT JT00 $3 0 | tail -1 | sed -e "s/ --init RB0-F-s0 / --init $2 /" -e 's/ --seed 0$/ --seed 1/'; }
case ${1:-} in
check)  # the s1 reference must reproduce the stored BNC2K-L18-s1 command except tag; the arms differ from it in tag / init / steps only
  diff <(cmd $REF JTN-L18-s1 2000 | tr ' ' '\n') <(tr ' ' '\n' < $W/cmd/$REF.txt) && echo "reference command identical to bn.sh's"
  for a in $ARMS; do IFS=: read T P N <<< $a
    d=$(diff <(cmd $REF JTN-L18-s1 2000 | tr ' ' '\n') <(cmd $T $P $N | tr ' ' '\n') | grep -c '^[<>]'); echo "$T: $d differing lines vs $REF (tag, init, steps: <= 6)"
    [[ $d -le 6 ]] || exit 1; done ;;
submit)
  mkdir -p $W/pool $J/pool; rm -f $W/CONT_DONE; bash $ME check || die "command check"
  ids=(); cr=()
  for a in $ARMS; do IFS=: read T P N <<< $a
    [[ ! -e $CK/$T/ckpt-final.pt ]] || die "$T exists"
    mkdir -p $W/cmd; cmd $T $P $N > $W/cmd/$T.txt
    t=$(sub bnc-train-$T "${FT[@]}" -- bash $ME train $T) || exit 1
    for c in $(seq 0 $(($(python3 -c "import json; print(len(json.load(open('$AH/meta.json'))['chunks']))") - 1))); do
      cr+=($(TRIES=2 sub bnc-ar-$T-c$c --after $t "${RD[@]}" -- bash $JT aread $T off $c)) || exit 1; done
    for i in 0 1 2 3 4 5; do cr+=($(TRIES=2 sub bnc-held-$T-$i --after $t "${RD[@]}" -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1; done
    ids+=($t); done
  # PAI 281 (ACC + GAP) of the s1 reference: BNC2K-L18-s1 already exists, so its read starts now (d366 serve, NTV0-L18-s0's chunk lists)
  [[ -e $CK/$REF/ckpt-final.pt ]] || die "no checkpoint $REF"
  mkdir -p $O/pai_accgap; [[ ! -e $O/pai_accgap/$REF ]] || die "PAI read of $REF exists"
  CHUNKS_FROM=$O/pai_accgap/NTV0-L18-s0/chunks SERVE="$PAISERVE" WIZ=controller.gains.rel_front_steering_angle_weight=10 CONC=4 PRIO=${PRIO:-12} \
    bash experiments/alpasim/scripts/pai_box.sh submit $O/pai_accgap/$REF $REF > $J/pai_$REF.submit.txt || die "pai submit"
  pp=$(awk '$2 ~ /^pai-/ {print $1}' $J/pai_$REF.submit.txt | paste -sd,); [[ -n $pp ]] || die "no PAI job"
  awk '$2 ~ /^pai-/' $J/pai_$REF.submit.txt >> $JOBS
  ps=$(sub bnc-paisum --after $pp --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME paisum) || exit 1
  s=$(sub bnc-report --after $(csv ${cr[@]}),$ps --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') submit: trains ${ids[*]} reads ${#cr[@]} PAI $pp paisum $ps report $s" | tee -a $W/stages.txt ;;
train)
  T=$2; [[ -f $W/cmd/$T.txt ]] || die "no command for $T"; echo "$(date '+%F %T') train $T: $(cat $W/cmd/$T.txt)" >> $W/stages.txt
  exec env ARCROWS=brake $(cat $W/cmd/$T.txt) ;;
paisum)
  mkdir -p $R/pai; P=experiments/defect_fix/scripts/pai281.py; RF=$DATA_DIR/runs/route_follow/acc_gap/read/s2
  .venv/bin/python $P --j $O/pai_accgap/$REF --jl "$REF+ACC+GAP" --j2 $O/pai_accgap/NTV0-L18-s0 --j2l "NTV0-L18-s0+ACC+GAP" --ref $O/pai_accgap/JTN-L18-s1 --rl "JTN-L18-s1+ACC+GAP" \
    --out $R/pai/pai281_vs_jtn_ntv0.md || die "pai281 vs jtn"
  .venv/bin/python $P --j $O/pai_accgap/$REF --jl "$REF+ACC+GAP" --ref $O/pai_accgap/NTV0-L18-s0 --rl "NTV0-L18-s0+ACC+GAP" --out $R/pai/pai281_vs_ntv0.md || die "pai281 vs ntv0"
  .venv/bin/python $P --j $O/pai_accgap/$REF --jl "$REF+ACC+GAP" --ref $RF --rl RSG+ACC+GAP --out $R/pai/pai281_vs_rsg.md || die "pai281 vs rsg"
  date '+%F %T' > $W/PAI_DONE ;;
report)
  [[ -f $W/PAI_DONE ]] || die "PAI summary not done"
  $TR $E/scripts/bnc.py || die "bnc.py rc $?"
  date '+%F %T' > $W/CONT_DONE ;;
wait)   # one blocker: until CONT_DONE, ERROR, or a failed / cancelled job of cont_jobs.txt
  while :; do
    [[ -f $W/CONT_DONE ]] && { echo CONT_DONE; exit 0; }
    [[ -f $W/ERROR ]] && { echo ERROR; cat $W/ERROR; exit 1; }
    f=$(.venv/bin/python - $JOBS <<'PY'
import sys
from jevdrive.cl import pool as P
ids = dict(l.split()[:2] for l in open(sys.argv[1]) if l.strip())
st = P.job_states(list(ids))
print(" ".join(f"{n}({i})" for i, n in ids.items() if st[i] in ("failed", "cancelled")))
PY
)
    [[ -n $f ]] && { echo "FAILED: $f"; exit 1; }
    sleep 300
  done ;;
*) sed -n 2,6p $0; exit 2 ;;
esac
