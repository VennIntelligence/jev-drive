#!/usr/bin/env bash
# PRACTICE-PAI (plans/2026-10-19-pai-practice-prereg.md): RSG-L18-s0 rolled on the practice-pai scenes with the PAI probe serve (frame dumps) -> S rows
# (onp_rows rows / cold -> onp_cache -> rf_pai_pool) -> input-path check R1 -> merged with RSG's pool on1_L18RB0 -> 1 000-step pilot and control from
# RSG-L18-s0 -> PAI 281 reads on the 4 fixed chunks -> the gate. Box, repo root; every job through the GPU pool at priority $PRIO (11); ids in $O/jobs.txt.
#   pp.sh chain        submit the whole chain at once (--after); the read-submit job submits the 8 reads and the gate job behind them
#   pp.sh dry          print the pilot and control training commands (DRY=1) and their diff (only --on-weight may differ)
#   job bodies: rows | cache | pool | fwd <rows> | r1 | clean | merge | train <pilot|control> | subreads | gate
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/practice_pai; ME=experiments/practice_pai/scripts/pp.sh; PY=experiments/practice_pai/scripts/pp.py; PRIO=${PRIO:-11}
TR=$DATA_DIR/envs/op-train/bin/python; AS=$DATA_DIR/third_party/alpasim; V=.venv/bin/python; S=experiments/route_follow/scripts
T=RSG-L18-s0; WRONG=RB0-F-s1; NP=PPAI160; NM=PPM160; CR=$DATA_DIR/runs/op_parity/cache
LIST=experiments/onpol_pai/results/split/train160.tsv                                   # = physical_ai_av/practice-pai@v1 (pp.py split checks it)
ROLLCH=$DATA_DIR/runs/route_follow/rsgacc/roll/pai/RSG-L18-s0/chunks                    # d333's 5 chunk lists of the same 160 scenes (scores comparable)
REF=$DATA_DIR/runs/route_follow/acc_gap/read/s2; GATE_CH="c000 c002 c004 c006"            # the reference read and the 4 fixed exam chunks (prereg)
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10   # the PAI probe serve
RSG="--warmup 1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-split alpasim_nuplan/onpol-train \
--lr-scale 0.1 --on-k 13 --on-mode self"                                                  # RSG-L18-s0's command minus --init / --on (= df.sh / jt.sh RSG)
declare -A TAG=([pilot]=PPAIP-L18-s0 [control]=PPAIC-L18-s0)
ROWS=$O/rows; STG=$DATA_DIR/tmp/practice_pai_stage; J=$(( $(nproc) > 24 ? 24 : $(nproc) ))
mkdir -p $O/pool
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner practice-pai --name "$name" --tries ${TRIES:-1} --priority $PRIO --no-check \
     --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
cmd_train() {  # cmd_train <arm> <weights>: the training command of an arm
  echo $TR experiments/route_mem/scripts/rm_train.py train --tag ${TAG[$1]} --init $T --allow-init $RSG --on on1_$NM --on-weight $2 --steps 1000 --seed 0; }
case ${1:-} in
dry)
  diff <(cmd_train pilot "pool=p:W" | tr ' ' '\n') <(cmd_train control "pool=p:0.000000" | tr ' ' '\n'); echo "pilot:   $(cmd_train pilot 'pool=p:W')"
  echo "control: $(cmd_train control 'pool=p:0.000000')" ;;
chain)
  for x in $O/roll/$T/jobs.txt $CR/on1_$NP $CR/on1_$NM $DATA_DIR/runs/op_parity/runs/${TAG[pilot]} $DATA_DIR/runs/op_parity/runs/${TAG[control]}; do
    [[ ! -e $x ]] || die "$x exists: never overwritten"; done
  CHUNKS_FROM=$ROLLCH SERVE="$SERVE PAI_DUMP=100000" WIZ=$WIZ KEEP_ASL=1 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $O/roll/$T $T > $O/roll_submit.txt || die "roll submit"
  n=$(ls $O/roll/$T/chunks/*.txt | wc -l); [[ $(wc -l < $O/roll/$T/jobs.txt) == $n ]] || die "not every roll chunk submitted"
  q=$(awk '{print $1}' $O/roll/$T/jobs.txt | paste -sd,); cat $O/roll/$T/jobs.txt >> $O/jobs.txt
  r=$(sub pp-rows-$NP --after $q --vram 0.5 --cpu 24 --ram 96 --timeout-h 3 -- bash $ME rows) || exit 1
  c=$(sub pp-cache-$NP --after $r --vram 20 --cpu 8 --ram 64 --timeout-h 2 -- bash $ME cache) || exit 1
  p=$(sub pp-pool-$NP --after $c --vram 0.5 --cpu 8 --ram 64 --timeout-h 2 -- bash $ME pool) || exit 1
  f1=$(sub pp-fwd-$NP-train --after $p --vram 16 --cpu 8 --ram 96 --timeout-h 2 -- bash $ME fwd train-rec) || exit 1
  f2=$(sub pp-fwd-$NP-dev --after $p --vram 16 --cpu 8 --ram 96 --timeout-h 2 -- bash $ME fwd dev-rec) || exit 1
  g=$(sub pp-r1-$NP --after $f1,$f2 --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- bash $ME r1) || exit 1
  sub pp-clean-$NP --after $g --vram 0.1 --cpu 2 --ram 4 --timeout-h 1 -- bash $ME clean > /dev/null || exit 1
  m=$(sub pp-merge-$NM --after $g --vram 0.1 --cpu 4 --ram 48 --timeout-h 1 -- bash $ME merge) || exit 1
  t=(); for A in pilot control; do
    t+=($(sub pp-train-$A-s0 --after $m --train --vram 30 --cpu 6 --ram 40 --env CL_HIST_KEY=rm-train --timeout-h 2 -- bash $ME train $A)) || exit 1; done
  s=$(sub pp-subreads --after $(csv ${t[@]}) --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $ME subreads) || exit 1
  echo "$(date '+%F %T') chain: roll $q rows $r cache $c pool $p fwd $f1 $f2 r1 $g merge $m train ${t[*]} subreads $s" | tee -a $O/stages.txt ;;
rows)
  mkdir -p $STG $ROWS
  env PAI_DUMP=100000 JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD $AS/.venv/bin/python experiments/onpol_pai/scripts/onp_rows.py rows --pass $NP \
      --runs "$O/roll/$T/runs/*" --out $ROWS --stage $STG --tag $T --jobs $J || exit 1
  env PAI_DUMP=100000 JEV_REPO=$PWD PYTHONPATH=$PWD/experiments/alpasim/lib:$PWD $TR experiments/onpol_pai/scripts/onp_rows.py cold --pass $NP --out $ROWS --stage $STG --jobs $J || exit 1
  env PYTHONPATH=$PWD/experiments/alpasim/lib:$PWD $TR experiments/onpol_pai/scripts/onp_rows.py report --pass $NP --out $ROWS > $ROWS/report_$NP.log 2>&1; tail -3 $ROWS/report_$NP.log ;;
cache) $TR experiments/onpol_pai/scripts/onp_cache.py build --pass $NP --rows $ROWS/$NP --out $O/cache ;;
pool)  $TR $S/rf_pai_pool.py --name $NP --rows $ROWS/$NP --cache $O/cache/$NP --runs "$O/roll/$T/runs/*" --split $LIST ;;
fwd)   $TR $S/rf_recov.py fwd --pool $NP --rows $2 --tags $T,$WRONG --out $O/fwd ;;
r1)    $TR $PY r1 --pool $NP --cache $O/cache/$NP --fwd $O/fwd --tag $T --control $WRONG --out $O/gates/r1_$NP.json ;;
clean)
  n=$(find $O/roll/$T/runs -name rollout.asl | wc -l); find $O/roll/$T/runs -name rollout.asl -delete; rm -rf $O/roll/$T/runs/*/sim/driver-logs/dump $STG
  echo "removed $n rollout.asl and the frame dumps; $(du -sh $O/roll | cut -f1) left" ;;
merge)
  $TR $S/rf_pait.py merge --pools on1_L18RB0:s:self on1_$NP:p:self --out $NM || exit 1
  for A in pilot control; do $TR $PY weights --pool $NM --arm $A > $O/w_$A.txt || exit 1; done; $TR experiments/route_mem/scripts/rm_train.py draws --on on1_$NM --on-weight "$(cat $O/w_pilot.txt)" ;;
train)
  A=$2; W=$(cat $O/w_$A.txt); [[ -n $W ]] || die "no weights for $A"; CMD=($(cmd_train $A $W))
  [[ -z ${DRY:-} ]] || { echo "${CMD[*]}"; exit 0; }
  exec "${CMD[@]}" ;;
subreads)  # the reads need the checkpoints at submit time (pai_box.sh checks them): submitted here, after both trainings, then the gate after them
  mkdir -p $O/chunks4; for c in $GATE_CH; do cp $REF/chunks/$c.txt $O/chunks4/; done; q=()
  for A in pilot control; do
    CHUNKS_FROM=$O/chunks4 SERVE="$SERVE" WIZ=$WIZ CONC=4 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $O/read/${TAG[$A]} ${TAG[$A]} > $O/read_$A.txt || die "read submit $A"
    [[ $(wc -l < $O/read/${TAG[$A]}/jobs.txt) == 4 ]] || die "$A: not 4 read chunks"; cat $O/read/${TAG[$A]}/jobs.txt >> $O/jobs.txt
    q+=($(awk '{print $1}' $O/read/${TAG[$A]}/jobs.txt)); done
  g=$(sub pp-gate --after $(csv ${q[@]}) --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME gate) || exit 1
  echo "$(date '+%F %T') subreads: reads ${q[*]} gate $g" | tee -a $O/stages.txt ;;
gate)
  $V $PY gate --pilot $O/read/${TAG[pilot]} --control $O/read/${TAG[control]} --ref $REF --chunks $(echo $GATE_CH | tr ' ' ,) --out $O/gate || { echo "gate crashed" > $O/ERROR; exit 1; }
  $V experiments/defect_fix/scripts/pai281.py --j $O/read/${TAG[pilot]} --jl pilot --j2 $O/read/${TAG[control]} --j2l control --ref $REF --rl RSG+ACC+GAP \
      --out $O/gate/pai281_4ch.md > /dev/null || { echo "pai281 crashed" > $O/ERROR; exit 1; }
  date '+%F %T' > $O/DONE ;;
*) sed -n 2,8p "$0"; exit 2 ;;
esac
