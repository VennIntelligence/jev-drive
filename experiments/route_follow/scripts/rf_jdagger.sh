#!/usr/bin/env bash
# JOINT-DAGGER round (plans/2026-10-16-joint-dagger-prereg.md): the rule expert's stop rule, joint DAgger labels on RSG-L18-s0's own states,
# dose pilots, full runs with staged reads. Box, repo root; every job through the GPU pool, --after chained; ids in $JD/jobs.txt.
#   rf_jdagger.sh prep          the whole preparation in one submit: stop files (held pack + the d319 stage packs) -> expert held read with the
#                               stop rule (rf_expert.sh box-read --expert-stop) -> xpgate; labels -> pools -> reference forward reads; the 60-step
#                               identity runs (RSG-L18 recipe vs pool on1_JDG at DAgger weight 0) -> bitcmp
#   rf_jdagger.sh packj         Amendment 1: one CPU job building on1_JDGJ (S rows + the junction / failure-window DAgger rows) and flags_j{10,15}
#   rf_jdagger.sh pilot <share> <lr>     1 000-step pilot JDGP<share>L<lr>-L18-s0 (share 25 | 50 % of the 128-row batch, or j10 | j15 = Amendment 1's
#                               subset at 10 | 15 %: tag JDGPJ10L03-...; --lr-scale lr) -> forward reads -> gate (j*: --sub; on PASS the gate job
#                               submits the held 460 @18 read + its report itself)
#   rf_jdagger.sh full <share> <lr>      JDG<SHARE>L<lr>-L18-s{0,1} 2 500 steps (+ X5K=1: JDG<share>L<lr>x5k-L18-s0, 5 000 steps) -> held 460 ->
#                               report -> stage2 (public guard + PAI 281 seed 0 only if H1 passes) (rf_recov_post.sh submit-held, STAGE2=1)
#   rf_jdagger.sh train <share|w0|off> <lr> <seed> <steps> <tag>      (job body)
# Env: PRIO (pool priority, default 8), AFTER (ids gating prep's first jobs / pilot / full)
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/data}
RF=$DATA_DIR/runs/route_follow; JD=$RF/joint_dagger; CK=$DATA_DIR/runs/op_parity/runs
TR=$DATA_DIR/envs/op-train/bin/python; ME=experiments/route_follow/scripts/rf_jdagger.sh; PY=experiments/route_follow/scripts/rf_jdagger.py
RC=experiments/route_follow/scripts/rf_recov.py; XS=experiments/route_follow/scripts/rf_expert.sh
P=$DATA_DIR/third_party/alpasim/.venv/bin/python
REC="--init RB0-F-s0 --warmup 1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-split alpasim_nuplan/onpol-train"
RB="--on on1_L18RB0 --on-k 13 --on-mode self"
mkdir -p $JD/pool
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner joint-dagger --name "$name" --tries 1 --priority ${PRIO:-8} --no-check \
     --log-dir $JD/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $JD/jobs.txt; echo "$id"; }
FW=(--train --vram 24 --cpu 6 --ram 70)
lrtag() { echo ${1//./}; }                                           # 0.3 -> 03, 1.0 -> 10
case ${1:-} in
prep)
  s=$(sub jdg-stops --vram 0.5 --cpu 24 --ram 48 --timeout-h 1 ${AFTER:+--after $AFTER} -- bash -c \
      "$TR $PY stops --pack $RF/rule_expert/pack_held --out $JD/stops/held && $TR $PY stops --pack $RF/rewind_onpol_all/rest/pack,$RF/rewind_onpol_all/s50/pack,$RF/rewind_onpol_all/smoke/pack,$RF/rewind_train64/pack --out $JD/stops/train") || exit 1
  AFTER=$s PRIO=${PRIO:-8} bash $XS box-read --expert-stop $JD/stops/held > $JD/xpread.txt || exit 1
  xr=$(awk '$2 ~ /^xp-read-/ {print $1}' $JD/xpread.txt | paste -sd,)
  xg=$(sub jdg-xpgate --after $xr --vram 0.5 --cpu 16 --ram 32 --timeout-h 1 -- bash $ME xpgate) || exit 1
  l=$(sub jdg-label --after $s --vram 0.5 --cpu 24 --ram 96 --timeout-h 1 -- $TR $PY label --workers 24) || exit 1
  k=$(sub jdg-pack --after $l --vram 0.5 --cpu 8 --ram 96 --timeout-h 1 -- $TR $PY pack) || exit 1
  f1=$(sub jdg-fwd-base --after $k --vram 20 --cpu 6 --ram 64 --timeout-h 1 -- bash -c \
       "$TR $RC fwd --pool JDGdev --rows dev-rec --tags RSG-L18-s0 && $TR $RC fwd --pool JDGd --rows train-rec --tags RSG-L18-s0") || exit 1
  g=(); for x in off w0; do g+=($(sub jdg-g-$x "${FW[@]}" --after $k --timeout-h 1 -- bash $ME train $x 0.1 0 60 JDGG-$x)) || exit 1; done
  b=$(sub jdg-bitcmp --after "$(IFS=,; echo "${g[*]}")" --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR $PY bitcmp) || exit 1
  echo "stops $s xpread $(awk '{print $1}' $JD/xpread.txt | paste -sd,) xpgate $xg label $l pack $k fwd-base $f1 gates ${g[*]} bitcmp $b" | tee -a $JD/prep.txt ;;
packj)
  k=$(sub jdg-packj --vram 0.5 --cpu 8 --ram 96 --timeout-h 1 ${AFTER:+--after $AFTER} -- $TR $PY packj) || exit 1
  echo "packj $k" | tee -a $JD/prep.txt ;;
pilot)   # pilot <share> <lr>
  SH=$2; LR=$3; T=JDGP${SH^^}L$(lrtag $LR)-L18-s0; GS=; [[ $SH == j* ]] && GS=--sub
  [[ ! -e $CK/$T/ckpt-final.pt ]] || { echo "$T exists" >&2; exit 1; }
  t=$(sub jdg-pilot-$T "${FW[@]}" --timeout-h 1 ${AFTER:+--after $AFTER} -- bash $ME train $SH $LR 0 1000 $T) || exit 1
  f=$(sub jdg-fwd-$T --after $t --vram 20 --cpu 6 --ram 64 --timeout-h 1 -- bash -c \
      "$TR $RC fwd --pool L18RSGallheld --rows held01 --tags $T && $TR $RC fwd --pool JDGdev --rows dev-rec --tags $T && $TR $RC fwd --pool JDGd --rows train-rec --tags $T") || exit 1
  gg=$(sub jdg-gate-$T --after $f --vram 0.5 --cpu 4 --ram 32 --timeout-h 1 --env PRIO=${PRIO:-8} -- $TR $PY gate --tag $T $GS --submit-held) || exit 1
  echo "$T: train $t fwd $f gate $gg" | tee -a $JD/pilots.txt ;;
full)    # full <share> <lr>
  SH=$2; LR=$3; A=JDG${SH^^}L$(lrtag $LR); ids=()
  for s in 0 1; do T=$A-L18-s$s; [[ ! -e $CK/$T/ckpt-final.pt ]] || { echo "$T exists" >&2; exit 1; }
    ids+=($(sub jdg-train-$T "${FW[@]}" --timeout-h 2 ${AFTER:+--after $AFTER} -- bash $ME train $SH $LR $s 2500 $T)) || exit 1; done
  ARMROOT=$JD KIND=J RESSUB=joint_dagger STAGE2=1 PRIO=${PRIO:-8} bash experiments/route_follow/scripts/rf_recov_post.sh submit-held $A $A-L18-s0 $A-L18-s1 ${ids[0]} ${ids[1]} || exit 1
  if [[ ${X5K:-0} == 1 ]]; then T=${A}x5k-L18-s0
    x=$(sub jdg-train-$T "${FW[@]}" --timeout-h 3 ${AFTER:+--after $AFTER} -- bash $ME train $SH $LR 0 5000 $T) || exit 1
    ARMROOT=$JD KIND=J RESSUB=joint_dagger STAGE2=1 PRIO=${PRIO:-8} bash experiments/route_follow/scripts/rf_recov_post.sh submit-held ${A}x5k $T - $x - || exit 1; fi
  echo "$A: trainings ${ids[*]} ${x:-}" | tee -a $JD/full.txt ;;
xpgate)  # the expert's held read with the stop rule (ST) against the stored one (E46): paired held confirm, rf_follow turns, xpgate summary
  M=$DATA_DIR/runs/route_mem/read; X=$JD/xpgate; mkdir -p $X; a=RSG-L18-s0-XP-L12-E46-ST; b=RSG-L18-s0-XP-L12-E46
  da=$(ls -d $M/$a/c*/run | paste -sd,); db=$(ls -d $M/$b/c*/run | paste -sd,)
  $P experiments/onpol_long/scripts/ol_confirm.py --runs "$a@held18=$da" "$b@held18=$db" --base $b --pairs "$a:$b" --out $X/held18 > $X/held18.log 2>&1 || exit 1
  $P experiments/route_follow/scripts/rf_follow.py --runs "$a@18=$da" "$b@18=$db" --out $X/follow --workers 16 > $X/follow.log 2>&1 || exit 1
  exec $P $PY xpgate --dir $X ;;
train)   # train <share|w0|off> <lr> <seed> <steps> <tag>
  case $2 in off) F="$RB" ;; *) f=$JD/flags_$2.txt; [[ -s $f ]] || { echo "no $f (rf_jdagger.py pack)" >&2; exit 1; }; F=$(cat $f) ;; esac
  exec $TR experiments/route_mem/scripts/rm_train.py train --tag $6 $REC --lr-scale $3 $F --steps $5 --seed $4 ;;
*) sed -n 2,14p $0; exit 2 ;;
esac
