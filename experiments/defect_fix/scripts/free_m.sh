#!/usr/bin/env bash
# NR1FREE-M (NR1 prereg supplement 16): the one-sided longitudinal term on moving states only (lib/nr1_free.py, rm_train --on-free W
# --on-free-mode move --on-free-frac 0.95). Box, repo root; every job through the GPU pool, ids in $F/jobs.txt. Env PRIO (10; reads PRIO + 1).
#   free_m.sh gate                 offline gate (free_m.py gate, CPU) -> $F/gate/gate.json, results/free_m/gate.json
#   free_m.sh identity <old rev>   JTN's command 60 steps eager, <old rev> vs this code (no --on-free) -> results/identity_free_m.json
#   free_m.sh pilot <after ids>    FREEM1K-L18-s0 (= JTNS1K-L18-s0's command + the term) and its reads, the control JTNS1K-L18-s0's arena subset
#                                  and PAI half; the subreads job submits the pilot's PAI half and the summary, which submits `full` on PASS
#   free_m.sh full                 FREEM-L18-s0 (= JTN-L18-s0's command + the term, 9 500 steps) -> native arena held 18 chunks + PAI 281
#   job bodies: train <tag> <steps> <warm> | psub | psum | fsub | fsum
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/defect_fix; F=$O/free_m; J=$O/jt; ME=experiments/defect_fix/scripts/free_m.sh; PY=experiments/defect_fix/scripts/free_m.py
TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; V=.venv/bin/python; JT=experiments/defect_fix/scripts/jt.sh
LAT=$O/lat/L18NR1v2; RES=experiments/defect_fix/results/free_m; W=${W:-0.004}; TERM="--on-free $W --on-free-mode move --on-free-frac 0.95"
T=FREEM1K-L18-s0; CTL=JTNS1K-L18-s0; TF=FREEM-L18-s0; TF0=JTN-L18-s0; SUB="0 3 6 9 12 15"
REF=$DATA_DIR/runs/route_follow/acc_gap/read/s2; PCH="c000 c002 c004 c006"
PAISERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10
AH=$DATA_DIR/runs/alpasim_proxy/arena/hold; RD=(--vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4)
mkdir -p $F/pool $RES
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner defect-fix --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-10} \
     --no-check --log-dir $F/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $F/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
pai() {  # pai <tag> <out> <chunks dir> [after]: PAI read with the d366 serve -> prints the job ids
  CHUNKS_FROM=$3 SERVE="$PAISERVE" WIZ=$WIZ CONC=4 PRIO=$((${PRIO:-10} + 1)) AFTER=${4:-} bash experiments/alpasim/scripts/pai_box.sh submit $2 $1 > $F/pai_$1.submit.txt \
    || die "pai submit $1"; awk '$2 ~ /^pai-/' $F/pai_$1.submit.txt >> $F/jobs.txt; awk '$2 ~ /^pai-/ {print $1}' $F/pai_$1.submit.txt | paste -sd,; }
case ${1:-} in
gate)
  g=$(sub freem-gate --vram 0.1 --cpu 24 --ram 96 --timeout-h 3 -- bash -c "$AP $PY gate --out $F/gate; rc=\$?; cp $F/gate/gate.json $RES/gate.json; exit \$rc") || exit 1
  echo "$(date '+%F %T') gate: $g" | tee -a $F/stages.txt; echo "$g" ;;
identity)
  rev=${2:?old rev}; WT=$DATA_DIR/tmp/df_old_$rev; [[ -d $WT ]] || git worktree add --detach $WT $rev > /dev/null || exit 1
  for X in FREEMGG-old FREEMGG-new; do [[ ! -e $DATA_DIR/runs/op_parity/runs/$X/ckpt-final.pt ]] || die "$X exists"; done
  FE=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train --env JEV_RM_EAGER=1 --env ARCROWS=brake); h=$(git rev-parse --short HEAD)
  a=$(sub freem-gg-old "${FE[@]}" --timeout-h 1 -- bash $WT/$JT train FREEMGG-old on1_JT $LAT JT00 60 0) || exit 1
  b=$(sub freem-gg-new "${FE[@]}" --timeout-h 1 -- bash $JT train FREEMGG-new on1_JT $LAT JT00 60 0) || exit 1
  e=$(sub freem-bitcmp --after $a,$b --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR experiments/defect_fix/scripts/df.py bitcmp --a FREEMGG-old \
      --b FREEMGG-new --out identity_free_m --note "JTN's command (JT00, arc 0, arc rows brake) 60 steps eager: code $rev vs $h without --on-free") || exit 1
  echo "$(date '+%F %T') identity: $a $b bitcmp $e" | tee -a $F/stages.txt; echo "$e" ;;
pilot)
  A=${2:?after ids}; [[ ! -e $DATA_DIR/runs/op_parity/runs/$T/ckpt-final.pt ]] || die "$T exists"
  [[ -e $DATA_DIR/runs/op_parity/runs/$CTL/ckpt-final.pt ]] || die "$CTL missing"
  mkdir -p $F/chunks4; for c in $PCH; do cp $REF/chunks/$c.txt $F/chunks4/; done
  t=$(sub freem-train-$T --after $A --train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train --timeout-h 2 -- bash $ME train $T 1000 50) || exit 1
  PR=$((${PRIO:-10} + 1)); ids=()
  r=$(PRIO=$PR bash experiments/alpasim_proxy/scripts/wait_set.sh readc $T $t | tail -1); [[ -n $r ]] || die "readc $T"; echo "$r ws-collect-$T" >> $F/jobs.txt; ids+=($r)
  for i in 0 1 2 3 4 5; do
    ids+=($(PRIO=$PR TRIES=2 sub freem-held-$T-$i --after $t "${RD[@]}" -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1; done
  ar=(); for c in $SUB; do
    ar+=($(PRIO=$PR TRIES=2 sub freem-ar-$T-c$c --after $t "${RD[@]}" -- bash $JT aread $T off $c)) || exit 1
    ar+=($(PRIO=$PR TRIES=2 sub freem-ar-$CTL-c$c --after $A "${RD[@]}" -- bash $JT aread $CTL off $c)) || exit 1; done
  pc=$(pai $CTL $F/pai/$CTL $F/chunks4 $A); [[ -n $pc ]] || die "control PAI"
  printf '%s\n' "${ids[*]} ${ar[*]}" "$(csv ${ar[@]})" "$pc" > $F/pilot_ids.txt
  s=$(sub freem-psub --after $t --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $ME psub) || exit 1
  echo "$(date '+%F %T') pilot: train $t wait set $r held/arena ${ids[*]:1} ${ar[*]} control PAI $pc subreads $s" | tee -a $F/stages.txt ;;
train)   # train <tag> <steps> <warm>: JTN's command (jt.sh train, ARCROWS=brake, draw JT00) + the term
  X=$2; N=$3; WM=$4
  CMD=$(DRY=1 ARCROWS=brake WARM=$WM bash $JT train $X on1_JT $LAT JT00 $N 0 | tail -1)
  [[ $CMD == *"--on-k 70 "* && $CMD == *"w/JT00.npz "* && $CMD == *"--warmup $WM "* && $CMD == *"--tag $X "* && $CMD == *"--steps $N "* ]] || die "unexpected command: $CMD"
  CMD="$CMD $TERM"; echo "$(date '+%F %T') train $X: $CMD" | tee -a $F/stages.txt
  exec $CMD ;;
psub)    # after the pilot's training: its PAI half, then extract / arena / PAI gate / summary behind every read
  { read -r ALL; read -r ARC; read -r PC; } < $F/pilot_ids.txt
  pp=$(pai $T $F/pai/$T $F/chunks4); [[ -n $pp ]] || die "pilot PAI"
  e=$(sub freem-ext-pilot --after $ARC --vram 0.1 --cpu 32 --ram 96 --timeout-h 2 -- $AP $PY ext --read $CTL=$J/arena/$CTL@off $T=$J/arena/$T@off --out $F/ext_pilot) || exit 1
  x=$(sub freem-arena-pilot --after $e --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- $V $PY arena --src $F/ext_pilot --a $T --b $CTL --out $F/arena_pilot.json) || exit 1
  g=$(sub freem-paigate --after $pp,$PC --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash -c "$V experiments/practice_pai/scripts/pp.py gate --pilot $F/pai/$T \
      --control $F/pai/$CTL --ref $REF --chunks $(echo $PCH | tr ' ' ,) --out $F/pai_gate; [[ -f $F/pai_gate/gate.json ]]") || exit 1   # its own FAIL exits 2: rule in psum
  s=$(sub freem-psum --after $(csv $ALL),$x,$g --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME psum) || exit 1
  echo "$(date '+%F %T') psub: pilot PAI $pp extract $e arena $x PAI gate $g summary $s" | tee -a $F/stages.txt ;;
psum)    # the pilot rule; PASS -> full
  cp $F/pai_gate/gate.md $RES/pai_half.md; cp $F/arena_pilot.json $RES/arena_pilot.json
  $TR $PY pilot --tag $T --ctl $CTL --arena $F/arena_pilot.json --pai $F/pai_gate/gate.json --out $RES/pilot.md | tee $F/pilot.txt || die "pilot summary"
  grep -q '^VERDICT PASS' $F/pilot.txt && { PRIO=${PRIO:-10} bash $ME full || die "full submit"; }
  date '+%F %T' > $F/PILOT_DONE ;;
full)
  [[ ! -e $DATA_DIR/runs/op_parity/runs/$TF/ckpt-final.pt ]] || die "$TF exists"
  t=$(sub freem-train-$TF --train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train --timeout-h 6 -- bash $ME train $TF 9500 500) || exit 1
  ar=(); nch=$(python3 -c "import json; print(len(json.load(open('$AH/meta.json'))['chunks']))")
  for c in $(seq 0 $((nch - 1))); do ar+=($(PRIO=$((${PRIO:-10} + 1)) TRIES=2 sub freem-ar-$TF-c$c --after $t "${RD[@]}" -- bash $JT aread $TF off $c)) || exit 1; done
  csv ${ar[@]} > $F/full_ids.txt
  s=$(sub freem-fsub --after $t --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $ME fsub) || exit 1
  echo "$(date '+%F %T') full: train $t arena ${ar[0]} .. ${ar[-1]} (${#ar[@]}) subreads $s" | tee -a $F/stages.txt ;;
fsub)
  ARC=$(cat $F/full_ids.txt); pp=$(pai $TF $F/pai281/$TF $O/pai_accgap/$TF0/chunks); [[ -n $pp ]] || die "full PAI"
  e=$(sub freem-ext-full --after $ARC --vram 0.1 --cpu 32 --ram 96 --timeout-h 3 -- $AP $PY ext --read $TF=$J/arena/$TF@off $TF0=$J/arena/$TF0@off \
      JTN-L18-s1=$J/arena/JTN-L18-s1@off RSG-L18-s0=$DATA_DIR/runs/alpasim_proxy/arena/reads/RSG-L18-s0 --out $F/ext_full) || exit 1
  x=$(sub freem-arena-full --after $e --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- $V $PY arena --src $F/ext_full --a $TF --b $TF0 --ref JTN-L18-s1 RSG-L18-s0 \
      --out $F/arena_full.json) || exit 1
  s=$(sub freem-fsum --after $x,$pp --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME fsum) || exit 1
  echo "$(date '+%F %T') fsub: PAI $pp extract $e arena $x summary $s" | tee -a $F/stages.txt ;;
fsum)
  cp $F/arena_full.json $RES/arena_full.json
  $V experiments/defect_fix/scripts/pai281.py --j $F/pai281/$TF --jl "$TF+ACC+GAP" --j2 $O/pai_accgap/$TF0 --j2l "$TF0+ACC+GAP" \
    --ref $REF --rl RSG+ACC+GAP --out $RES/pai281_accgap.md || die "pai281"
  date '+%F %T' > $F/FULL_DONE ;;
*) sed -n 2,9p $0; exit 2 ;;
esac
