#!/usr/bin/env bash
# WAIT-TRAIN chain (plans/2026-10-17-wait-train-prereg.md). Box, repo root; every job through the GPU pool, --after chained; ids in $O/jobs.txt.
#   wt.sh rows                  one submit: select (+ leak check, split) -> index -> pp_prep W -> route memory -> teacher -> pack (on1_WAIT, merged
#                               on1_L18RBW, flags) -> the 60-step identity runs WTGG-off (RSG-L18 recipe) / WTGG-w0 (merged pool, new rows weight 0) -> bitcmp
#   wt.sh pilot                 WTJ2P / WTJ5P / WTCP -L18-s0, 1 000 steps -> waitcap_test plans + dev plans -> gate (on PASS submits `full <dose>`)
#   wt.sh full <j2|j5>          WTJ{2,5}-L18-s{0,1} and WTC-L18-s{0,1}, 2 500 steps -> plans -> held 460 @18 (6 chunks each) -> clgate held
#   wt.sh stage-sub J0,J1 C0,C1 the public static scene @10 (4 tags) + PAI stop17 (J0, submission config) -> clgate sub
#   wt.sh stage-pub J0,J1 C0,C1 public 1491 @10 (10 chunks each) -> clgate pub
#   wt.sh stage-pai J0,J1 C0,C1 PAI 281, J0, submission config (gains 10 + JEV_ACC=1, the reference's 8 chunk lists) -> clgate pai
#   wt.sh train <off|w0|j2|j5> <seed> <steps> <tag>      (job body)
# Env: PRIO (pool priority, default 9)
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}
O=$DATA_DIR/runs/wait_cap/train; CK=$DATA_DIR/runs/op_parity/runs; ME=experiments/wait_cap/scripts/wt.sh; PY=experiments/wait_cap/scripts/wt.py
TR=$DATA_DIR/envs/op-train/bin/python; NS=$DATA_DIR/envs/navsim2/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
WC=experiments/wait_cap/scripts/wc.py; OL=$DATA_DIR/runs/onpol_long; D=idx_waitcap_train
REC="--init RB0-F-s0 --warmup 1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-split alpasim_nuplan/onpol-train"
RB="--on on1_L18RB0 --on-k 13 --on-mode self"
PAICH=$DATA_DIR/runs/alpasim/pai_box/tokyo/r281/rsg/chunks
PAISERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1"; PAIWIZ=controller.gains.rel_front_steering_angle_weight=10
STATIC=2021.09.16.15.12.03_veh-42_01037_01434-b5b8d20688385790
STOP17="054b5901 095cf563 127c4f85 2d10a08d 35e082a4 3fe2bac8 44df49fe 4cb7f7ac 4ecc114d 559d4ad6 68c3be44 7d994ed3 9f8acd00 ba40be62 bc309fd6 cbd87234 fc9fde87"
mkdir -p $O/pool
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner wait-train --name "$name" --tries 1 --priority ${PRIO:-9} --no-check \
     --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
FW=(--train --vram 24 --cpu 6 --ram 70)
csv() { local IFS=,; echo "$*"; }
case ${1:-} in
rows)
  s=$(sub wt-select --vram 0.5 --cpu 16 --ram 48 --timeout-h 1 -- $TR $PY select --workers 16) || exit 1
  i=$(sub wt-index --after $s --vram 0.5 --cpu 4 --ram 32 --timeout-h 2 -- $NS $PY index) || exit 1
  p=$(sub wt-prep --after $i --vram 10 --cpu 12 --ram 48 --timeout-h 2 -- $TR experiments/op_parity/scripts/pp_prep.py --data $D --frames warp --no-side --synth cpu --workers 12) || exit 1
  r=$(sub wt-rmnav --after $p --vram 0.5 --cpu 12 --ram 40 --timeout-h 1 -- $AP experiments/route_mem/scripts/rm_build.py nav --data $D --workers 12) || exit 1
  t=$(sub wt-teacher --after $p --vram 10 --cpu 4 --ram 32 --timeout-h 1 -- $TR $PY teacher) || exit 1
  k=$(sub wt-pack --after $r,$t --vram 0.5 --cpu 4 --ram 96 --timeout-h 1 -- $TR $PY pack) || exit 1
  g=(); for x in off w0; do g+=($(sub wt-gg-$x "${FW[@]}" --after $k --timeout-h 1 -- bash $ME train $x 0 60 WTGG-$x)) || exit 1; done
  b=$(sub wt-bitcmp --after "$(csv ${g[@]})" --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR $PY bitcmp) || exit 1
  echo "select $s index $i prep $p rmnav $r teacher $t pack $k identity ${g[*]} bitcmp $b" | tee -a $O/rows.txt ;;
pilot)
  [[ -s $O/identity.json ]] && grep -q '"identical": true' $O/identity.json || { echo "identity not proven ($O/identity.json)" >&2; exit 1; }
  ids=(); TT=(WTJ2P-L18-s0 WTJ5P-L18-s0 WTCP-L18-s0)
  for q in j2 j5 w0; do T=WT${q^^}P-L18-s0; [[ $q == w0 ]] && T=WTCP-L18-s0; [[ ! -e $CK/$T/ckpt-final.pt ]] || { echo "$T exists" >&2; exit 1; }
    ids+=($(sub wt-pilot-$T "${FW[@]}" --timeout-h 1 -- bash $ME train $q 0 1000 $T)) || exit 1; done
  pl=$(sub wt-plans-pilot --after "$(csv ${ids[@]})" --vram 10 --cpu 4 --ram 40 --timeout-h 1 -- $TR $WC plans --tags ${TT[@]}) || exit 1
  dp=$(sub wt-dev-pilot --after "$(csv ${ids[@]})" --vram 20 --cpu 6 --ram 64 --timeout-h 1 -- $TR $PY devplans --tags ${TT[@]}) || exit 1
  gg=$(sub wt-gate-pilot --after $pl,$dp --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 --env PRIO=${PRIO:-9} -- \
       $TR $PY gate --j WTJ2P-L18-s0 WTJ5P-L18-s0 --c WTCP-L18-s0 --dose-of j2=WTJ2P-L18-s0,j5=WTJ5P-L18-s0 --submit) || exit 1
  echo "pilot trainings ${ids[*]} plans $pl dev $dp gate $gg" | tee -a $O/stages.txt ;;
full)    # full <j2|j5>
  q=${2:?dose}; J=WT${q^^}-L18; tr=(); JJ=($J-s0 $J-s1); CC=(WTC-L18-s0 WTC-L18-s1)
  for s in 0 1; do for x in $q w0; do T=$J-s$s; [[ $x == w0 ]] && T=WTC-L18-s$s; [[ ! -e $CK/$T/ckpt-final.pt ]] || { echo "$T exists" >&2; exit 1; }
    tr+=($(sub wt-train-$T "${FW[@]}" --timeout-h 2 -- bash $ME train $x $s 2500 $T)) || exit 1; done; done
  pl=$(sub wt-plans-full --after "$(csv ${tr[@]})" --vram 10 --cpu 4 --ram 40 --timeout-h 1 -- $TR $WC plans --tags ${JJ[@]} ${CC[@]}) || exit 1
  dp=$(sub wt-dev-full --after "$(csv ${tr[@]})" --vram 20 --cpu 6 --ram 64 --timeout-h 1 -- $TR $PY devplans --tags ${JJ[@]} ${CC[@]}) || exit 1
  hh=(); k=0; for T in $J-s0 WTC-L18-s0 $J-s1 WTC-L18-s1; do
    for i in 0 1 2 3 4 5; do hh+=($(sub wt-held-$T-$i --after ${tr[$k]} --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
        -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1; done; k=$((k + 1)); done
  g=$(sub wt-clgate-held --after "$(csv ${hh[@]})" --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 --env PRIO=${PRIO:-9} -- \
      $TR $PY clgate --stage held --j $(csv ${JJ[@]}) --c $(csv ${CC[@]}) --submit) || exit 1
  echo "full $q: trainings ${tr[*]} plans $pl dev $dp held ${#hh[@]} jobs gate $g" | tee -a $O/stages.txt ;;
stage-sub)
  JS=${2:?J0,J1}; CS=${3:?C0,C1}; J0=${JS%%,*}; rr=(); mkdir -p $O/sub $O/paisub
  echo $STATIC > $O/sub/static.txt
  for T in ${JS//,/ } ${CS//,/ }; do rr+=($(sub wt-sub-$T --vram 20 --cpu 6 --ram 30 --timeout-h 1 --env CL_HIST_KEY=alpasim-read4 \
      -- bash experiments/onpol_long/scripts/ol_chain.sh readlist $T $O/sub/static.txt 10 $O/sub/$T/run)) || exit 1; done
  cat $PAICH/*.txt | grep -E "^clipgt-($(echo $STOP17 | tr ' ' '|'))" | sort -u > $O/paisub/stop17.txt
  (( $(wc -l < $O/paisub/stop17.txt) == 17 )) || { echo "stop17 list: $(wc -l < $O/paisub/stop17.txt) scenes" >&2; exit 1; }
  CHUNK=17 SERVE="$PAISERVE" WIZ=$PAIWIZ PRIO=${PRIO:-9} bash experiments/alpasim/scripts/pai_box.sh submit $O/paisub/$J0 $J0 $O/paisub/stop17.txt > $O/paisub/submit.txt || exit 1
  pp=$(awk '{print $1}' $O/paisub/submit.txt | grep -E '^[0-9]{4}-' | paste -sd,)
  [[ -n $pp ]] || { echo "no PAI subset job" >&2; exit 1; }
  g=$(sub wt-clgate-sub --after "$(csv ${rr[@]}),$pp" --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 --env PRIO=${PRIO:-9} -- $TR $PY clgate --stage sub --j $JS --c $CS --submit) || exit 1
  echo "stage-sub: public static ${rr[*]} PAI stop17 $pp gate $g" | tee -a $O/stages.txt ;;
stage-pub)
  JS=${2:?}; CS=${3:?}; rr=()
  for T in ${JS//,/ } ${CS//,/ }; do [[ ! -e $O/read10/$T ]] || { echo "$O/read10/$T exists" >&2; exit 1; }
    for i in $(seq 0 9); do rr+=($(sub wt-g10-$T-$i --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
        -- bash experiments/onpol_long/scripts/ol_chain.sh readlist $T $OL/lists/all_n10/c$i.txt 10 $O/read10/$T/b$i)) || exit 1; done; done
  g=$(sub wt-clgate-pub --after "$(csv ${rr[@]})" --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 --env PRIO=${PRIO:-9} -- $TR $PY clgate --stage pub --j $JS --c $CS --submit) || exit 1
  echo "stage-pub: ${#rr[@]} read jobs gate $g" | tee -a $O/stages.txt ;;
stage-pai)
  JS=${2:?}; CS=${3:?}; J0=${JS%%,*}; [[ ! -e $O/pai/$J0 ]] || { echo "$O/pai/$J0 exists" >&2; exit 1; }
  CHUNKS_FROM=$PAICH SERVE="$PAISERVE" WIZ=$PAIWIZ PRIO=${PRIO:-9} bash experiments/alpasim/scripts/pai_box.sh submit $O/pai/$J0 $J0 > $O/pai_submit.txt || exit 1
  pp=$(awk '{print $1}' $O/pai_submit.txt | grep -E '^[0-9]{4}-' | paste -sd,)
  [[ -n $pp ]] || { echo "no PAI job" >&2; exit 1; }
  g=$(sub wt-clgate-pai --after $pp --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR $PY clgate --stage pai --j $JS --c $CS) || exit 1
  echo "stage-pai: $(echo $pp | tr , '\n' | wc -l) chunk jobs gate $g" | tee -a $O/stages.txt ;;
train)   # train <off|w0|j2|j5> <seed> <steps> <tag>
  case $2 in off) F="$RB" ;; *) f=$O/flags_$2.txt; [[ -s $f ]] || { echo "no $f (wt.py pack)" >&2; exit 1; }; F=$(cat $f) ;; esac
  PYTHONPATH=$PWD/experiments/wait_cap/lib${PYTHONPATH:+:$PYTHONPATH} exec $TR experiments/route_mem/scripts/rm_train.py train --tag $5 $REC --lr-scale 0.1 $F --steps $4 --seed $3 ;;
*) sed -n 2,11p $0; exit 2 ;;
esac
