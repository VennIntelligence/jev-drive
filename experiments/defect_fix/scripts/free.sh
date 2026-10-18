#!/usr/bin/env bash
# NR1FREE (NR1 prereg supplement 15; practice plan item 5): the one-sided longitudinal consequence term (lib/nr1_free.py, rm_train --on-free).
# Box, repo root; every job through the GPU pool, ids in $O/free/jobs.txt. Env PRIO (10; reads PRIO + 1).
#   free.sh gate                 offline gate (free_gate.py gate, CPU) -> $O/free/gate/gate.json, results/free/gate.json
#   free.sh identity <old rev>   JTN's command (JT00, arc 0, rows brake) 60 steps eager, <old rev> vs this code (no --on-free) -> df.py bitcmp
#                                -> results/identity_free.json
#   free.sh pilot [after]        1 000-step pilot NR1FREE1K-L18-s0 (= JTNS1K's command with the natural-share draw JT00N + --on-free W) and
#                                its control NTVN1K-L18-s0 (the same without --on-free) -> per arm wait set v2 (readc) + held 460 @18
#                                -> free_gate.py pilot -> results/free/pilot.md
#   free.sh train <tag> <W>      job body
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/defect_fix; F=$O/free; ME=experiments/defect_fix/scripts/free.sh; G=experiments/defect_fix/scripts/free_gate.py
TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; JT=experiments/defect_fix/scripts/jt.sh
LAT=$O/lat/L18NR1v2; RES=experiments/defect_fix/results/free; W=${W:-0.004}
mkdir -p $F/pool $RES
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner defect-fix --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-10} \
     --no-check --log-dir $F/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $F/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
case ${1:-} in
gate)
  g=$(sub free-gate --vram 0.1 --cpu 16 --ram 64 --timeout-h 2 -- bash -c "$AP $G gate --out $F/gate; rc=\$?; cp $F/gate/gate.json $RES/gate.json; exit \$rc") || exit 1
  echo "$(date '+%F %T') gate: $g" | tee -a $F/stages.txt ;;
identity)
  rev=${2:?old rev}; WT=$DATA_DIR/tmp/df_old_$rev; [[ -d $WT ]] || git worktree add --detach $WT $rev > /dev/null || exit 1
  for T in FREEGG-old FREEGG-new; do [[ ! -e $DATA_DIR/runs/op_parity/runs/$T/ckpt-final.pt ]] || die "$T exists"; done
  FE=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train --env JEV_RM_EAGER=1 --env ARCROWS=brake); h=$(git rev-parse --short HEAD)
  a=$(sub free-gg-old "${FE[@]}" --timeout-h 1 -- bash $WT/$JT train FREEGG-old on1_JT $LAT JT00 60 0) || exit 1
  b=$(sub free-gg-new "${FE[@]}" --timeout-h 1 -- bash $JT train FREEGG-new on1_JT $LAT JT00 60 0) || exit 1
  e=$(sub free-bitcmp --after $a,$b --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR experiments/defect_fix/scripts/df.py bitcmp --a FREEGG-old \
      --b FREEGG-new --out identity_free --note "JTN's command (JT00, arc 0, arc rows brake) 60 steps eager: code $rev vs $h without --on-free") || exit 1
  echo "$(date '+%F %T') identity: $a $b bitcmp $e" | tee -a $F/stages.txt; echo "$e" ;;
pilot)
  A=${2:-}; ids=()
  for x in NTVN1K-L18-s0:0 NR1FREE1K-L18-s0:$W; do T=${x%%:*}; WW=${x#*:}
    [[ ! -e $DATA_DIR/runs/op_parity/runs/$T/ckpt-final.pt ]] || die "$T exists"
    t=$(sub free-train-$T ${A:+--after $A} --train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train --timeout-h 1 -- bash $ME train $T $WW) || exit 1
    r=$(PRIO=$((${PRIO:-10} + 1)) bash experiments/alpasim_proxy/scripts/wait_set.sh readc $T $t | tail -1); [[ -n $r ]] || die "readc $T"
    echo "$r ws-collect-$T" >> $F/jobs.txt; ids+=($r)
    for i in 0 1 2 3 4 5; do
      ids+=($(PRIO=$((${PRIO:-10} + 1)) TRIES=2 sub free-held-$T-$i --after $t --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
          -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1; done
    echo "$(date '+%F %T') pilot arm $T (on-free $WW): train $t wait-set collect $r" | tee -a $F/stages.txt
  done
  s=$(sub free-pilot-sum --after $(csv ${ids[@]}) --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- $TR $G pilot --tag NR1FREE1K-L18-s0 --ctl NTVN1K-L18-s0 \
      --out $RES/pilot.md) || exit 1
  echo "$(date '+%F %T') pilot: reads ${ids[*]}; summary $s" | tee -a $F/stages.txt ;;
train)   # train <tag> <W>: JTNS1K-L18-s0's command (jt.sh train, ARCROWS=brake, WARM=50, 1 000 steps) with the draw JT00N, + --on-free W when W > 0
  T=$2; WW=$3
  CMD=$(DRY=1 ARCROWS=brake WARM=50 bash $JT train $T on1_JT $LAT JT00N 1000 0 | tail -1)
  [[ $CMD == *"--on-k 70 "* && $CMD == *"w/JT00N.npz "* && $CMD == *"--warmup 50 "* && $CMD == *"--tag $T "* ]] || die "unexpected command: $CMD"
  python3 -c "import sys; sys.exit(0 if float('$WW') > 0 else 1)" && CMD="$CMD --on-free $WW"
  echo "$(date '+%F %T') train $T: $CMD" | tee -a $F/stages.txt
  exec $CMD ;;
*) sed -n 2,11p $0; exit 2 ;;
esac
