#!/usr/bin/env bash
# STACK round (plans/2026-10-16-stack-prereg.md): identity gates, smoke, dose pilots, full runs with staged reads. Box, repo root; every job
# through the GPU pool, --after chained; ids in $SK/jobs.txt.
#   rf_stack.sh gates            STKG-off (60 steps, every STACK switch off = the RSG-L18 recipe on the new code) -> bitcmp against JDGG-off
#   rf_stack.sh rx0              the reflex module at its init adds exact zeros (rf_stack.py rx0; needs stack/self/on1_L18RSG.npz)
#   rf_stack.sh smoke <A|B>      STKS<lv> 60 steps with every switch of the arm on (flags_<lv>.txt): sizes of the terms, multipliers, speed
#   rf_stack.sh pilot <A|B> <lr> 1 000-step pilot STK<lv>L<lr>-L18-s0 -> forward reads (held decision 0 / 1 rows, dev own rows; the base's dev
#                                own rows if missing) -> gate (on PASS the gate job submits the held 460 @18 read + report, PRIO 9)
#   rf_stack.sh full <A|B> <lr>  STK<lv>L<lr>-L18-s{0,1} 2 500 steps -> held 460 -> report -> stage2 (public guard + PAI 281 seed 0 if H1 passes)
#   rf_stack.sh train <off|A|B> <lr> <seed> <steps> <tag>      (job body)
# Env: PRIO (default 8; held reads of passing pilots 9), AFTER (ids gating the first jobs)
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/data}
RF=$DATA_DIR/runs/route_follow; SK=$RF/stack; CK=$DATA_DIR/runs/op_parity/runs; RE=$RF/recovery_envelope
TR=$DATA_DIR/envs/op-train/bin/python; ME=experiments/route_follow/scripts/rf_stack.sh; PY=experiments/route_follow/scripts/rf_stack.py
RC=experiments/route_follow/scripts/rf_recov.py
REC="--init RB0-F-s0 --warmup 1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-split alpasim_nuplan/onpol-train"
RB="--on on1_L18RB0 --on-k 13 --on-mode self"
mkdir -p $SK/pool
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner stack --name "$name" --tries 1 --priority ${PRIO:-8} --no-check \
     --log-dir $SK/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $SK/jobs.txt; echo "$id"; }
FW=(--train --vram 30 --cpu 6 --ram 80)
lrtag() { echo ${1//./}; }                                           # 0.1 -> 01
case ${1:-} in
gates)
  [[ ! -e $CK/STKG-off/ckpt-final.pt ]] || { echo "STKG-off exists" >&2; exit 1; }
  g=$(sub stk-g-off "${FW[@]}" --timeout-h 1 ${AFTER:+--after $AFTER} -- bash $ME train off 0.1 0 60 STKG-off) || exit 1
  b=$(sub stk-bitcmp --after $g --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR $PY bitcmp) || exit 1
  echo "gates: train $g bitcmp $b" | tee -a $SK/prep.txt ;;
rx0)
  x=$(sub stk-rx0 --vram 30 --cpu 6 --ram 80 --timeout-h 1 ${AFTER:+--after $AFTER} -- $TR $PY rx0) || exit 1
  echo "rx0 $x" | tee -a $SK/prep.txt ;;
smoke)
  LV=$2; T=STKS$LV; [[ ! -e $CK/$T/ckpt-final.pt ]] || { echo "$T exists" >&2; exit 1; }
  s=$(sub stk-smoke-$LV "${FW[@]}" --timeout-h 1 ${AFTER:+--after $AFTER} -- bash $ME train $LV 0.1 0 60 $T) || exit 1
  echo "smoke $LV $s" | tee -a $SK/prep.txt ;;
pilot)   # pilot <A|B> <lr>
  LV=$2; LR=$3; T=STK${LV}L$(lrtag $LR)-L18-s0
  [[ ! -e $CK/$T/ckpt-final.pt ]] || { echo "$T exists" >&2; exit 1; }
  t=$(sub stk-pilot-$T "${FW[@]}" --timeout-h 2 ${AFTER:+--after $AFTER} -- bash $ME train $LV $LR 0 1000 $T) || exit 1
  fb=; if [[ ! -s $RE/fwd/L18RSG-dev-rec/plans_RSG-L18-s0.npz ]] && [[ ! -s $SK/fwdbase.id ]]; then
    fb=$(sub stk-fwd-base --vram 20 --cpu 6 --ram 64 --timeout-h 1 -- $TR $RC fwd --pool L18RSG --rows dev-rec --tags RSG-L18-s0) || exit 1; echo $fb > $SK/fwdbase.id
  elif [[ -s $SK/fwdbase.id ]]; then fb=$(cat $SK/fwdbase.id); fi
  f=$(sub stk-fwd-$T --after $t --vram 20 --cpu 6 --ram 64 --timeout-h 1 -- bash -c \
      "$TR $RC fwd --pool L18RSGallheld --rows held01 --tags $T && $TR $RC fwd --pool L18RSG --rows dev-rec --tags $T") || exit 1
  gg=$(sub stk-gate-$T --after $f${fb:+,$fb} --vram 0.5 --cpu 8 --ram 32 --timeout-h 1 --env PRIO=9 -- $TR $PY gate --tag $T --submit-held) || exit 1
  echo "$T: train $t fwd $f gate $gg" | tee -a $SK/pilots.txt ;;
full)    # full <A|B> <lr>
  LV=$2; LR=$3; A=STK${LV}L$(lrtag $LR); ids=()
  for s in 0 1; do T=$A-L18-s$s; [[ ! -e $CK/$T/ckpt-final.pt ]] || { echo "$T exists" >&2; exit 1; }
    ids+=($(sub stk-train-$T "${FW[@]}" --timeout-h 3 ${AFTER:+--after $AFTER} -- bash $ME train $LV $LR $s 2500 $T)) || exit 1; done
  ARMROOT=$SK KIND=S RESSUB=stack STAGE2=1 PRIO=${PRIO:-8} bash experiments/route_follow/scripts/rf_recov_post.sh submit-held $A $A-L18-s0 $A-L18-s1 ${ids[0]} ${ids[1]} || exit 1
  echo "$A: trainings ${ids[*]}" | tee -a $SK/full.txt ;;
train)   # train <off|A|B> <lr> <seed> <steps> <tag>
  case $2 in off) F="$RB" ;; *) f=$SK/flags_$2.txt; [[ -s $f ]] || { echo "no $f (rf_stack.py flags)" >&2; exit 1; }; F=$(cat $f) ;; esac
  exec $TR experiments/route_mem/scripts/rm_train.py train --tag $6 $REC --lr-scale $3 $F --steps $5 --seed $4 ;;
*) sed -n 2,11p $0; exit 2 ;;
esac
