#!/usr/bin/env bash
# NR1STAND (NR1 prereg supplement 17): the standing-row mirror term (lib/nr1_free.py stand_*, rm_train --on-stand W), alone (A) and with
# FREE-M at half dose (B). Box, repo root; every job through the GPU pool, ids in $S/jobs.txt. Env PRIO (10; reads PRIO + 1).
#   stand.sh check                 the DRY commands of both arms against JTNS1K-L18-s0's -> $S/cmd_*.txt, results/stand/cmd_diff.txt
#   stand.sh pre <old rev>         offline gate (CPU), switch-off identity (60 steps eager, <old rev> vs this code), dose (frozen init, 50 steps)
#                                  -> results/stand/{gate,identity,dose}.json; prints the three ids
#   stand.sh pilot <after ids>     STANDFR1K-L18-s0 (C, + d365's FREE at 0.004 / 0.8), STAND1K-L18-s0 (A), STANDFM1K-L18-s0 (B; env
#                                  ARMS_RUN picks a subset, order = priority): train (checks gate / identity / dose first), wait set v2, held 460
#                                  @18, arena subset (6 chunks); then extract, arena, drift, per-arm summary
#   job bodies: gate | dose | train <arm> | sum
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/defect_fix; S=$O/stand; J=$O/jt; ME=experiments/defect_fix/scripts/stand.sh; PY=experiments/defect_fix/scripts/stand.py
FM=experiments/defect_fix/scripts/free_m.py; TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; V=.venv/bin/python
JT=experiments/defect_fix/scripts/jt.sh; LAT=$O/lat/L18NR1v2; RES=experiments/defect_fix/results/stand; CTL=JTNS1K-L18-s0; SUB="0 3 6 9 12 15"
declare -A TAG=([A]=STAND1K-L18-s0 [B]=STANDFM1K-L18-s0 [C]=STANDFR1K-L18-s0) LAB=([A]=STAND [B]=STANDFM [C]=STANDFR) TPR=([A]=10 [B]=11 [C]=10)
ARMS="C A B"
FREEB="--on-free 0.002 --on-free-mode move --on-free-frac 0.95"; FREEC="--on-free 0.004"
RD=(--vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4)
mkdir -p $S/pool $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $S/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner defect-fix --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-10} \
     --no-check --log-dir $S/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $S/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
base() {  # base <tag> <steps> <warm>: JTN's command (jt.sh train DRY, ARCROWS=brake, draw JT00), checked
  local C; C=$(DRY=1 ARCROWS=brake WARM=$3 bash $JT train $1 on1_JT $LAT JT00 $2 0 | tail -1)
  [[ $C == *"--on-k 70 "* && $C == *"w/JT00.npz "* && $C == *"--warmup $3 "* && $C == *"--tag $1 "* && $C == *"--steps $2 "* && $C == *"--init RB0-F-s0 "* ]] \
    || die "unexpected command: $C"; echo "$C"; }
term() {  # term <arm> <W_S>
  local t="--on-stand $2 --on-stand-margin 1.0"; [[ $1 == B ]] && t="$t $FREEB"; [[ $1 == C ]] && t="$t $FREEC"; echo "$t"; }
case ${1:-} in
check)
  W=${W_S:-0.001}; base $CTL 1000 50 > $S/cmd_ctl.txt || exit 1
  for A in $ARMS; do echo "$(base ${TAG[$A]} 1000 50) $(term $A $W)" > $S/cmd_$A.txt || exit 1; done
  { for A in $ARMS; do echo "== $CTL -> ${TAG[$A]} (W_S $W)"; diff <(tr ' ' '\n' < $S/cmd_ctl.txt) <(tr ' ' '\n' < $S/cmd_$A.txt); done; } > $RES/cmd_diff.txt
  cat $RES/cmd_diff.txt
  declare -A NW=([A]=5 [B]=11 [C]=7)
  for A in $ARMS; do n=$(diff <(tr ' ' '\n' < $S/cmd_ctl.txt) <(tr ' ' '\n' < $S/cmd_$A.txt) | grep -c '^>'); want=${NW[$A]}
    [[ $n == "$want" ]] || die "arm $A: $n new tokens, want $want (--tag + the switches)"; done; echo "diff ok" ;;
pre)
  rev=${2:?old rev}; WT=$DATA_DIR/tmp/df_old_$rev; [[ -d $WT ]] || git worktree add --detach $WT $rev > /dev/null || exit 1
  for X in STANDGG-old STANDGG-new STANDDOSE-L18-s0; do [[ ! -e $DATA_DIR/runs/op_parity/runs/$X/ckpt-final.pt ]] || die "$X exists"; done
  g=$(sub stand-gate --vram 0.1 --cpu 24 --ram 96 --timeout-h 3 -- bash $ME gate) || exit 1
  FE=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train --env JEV_RM_EAGER=1 --env ARCROWS=brake); h=$(git rev-parse --short HEAD)
  a=$(sub stand-gg-old "${FE[@]}" --timeout-h 1 -- bash $WT/$JT train STANDGG-old on1_JT $LAT JT00 60 0) || exit 1
  b=$(sub stand-gg-new "${FE[@]}" --timeout-h 1 -- bash $JT train STANDGG-new on1_JT $LAT JT00 60 0) || exit 1
  e=$(sub stand-bitcmp --after $a,$b --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR experiments/defect_fix/scripts/df.py bitcmp --a STANDGG-old \
      --b STANDGG-new --out stand/identity --note "JTN's command (JT00, arc 0, arc rows brake) 60 steps eager: code $rev vs $h without --on-stand") || exit 1
  d=$(sub stand-dose --train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME dose) || exit 1
  echo "$(date '+%F %T') pre: gate $g identity $a $b bitcmp $e dose $d" | tee -a $S/stages.txt; echo "$g,$e,$d" ;;
gate)
  $AP $PY gate --out $S/gate; rc=$?; cp $S/gate/gate.json $RES/gate.json; exit $rc ;;
dose)    # frozen init (lr 0), both terms on, per-term trunk gradients every step -> W_S
  X=STANDDOSE-L18-s0; CMD="$(base $X 50 50) --on-free 0.004 --on-free-mode move --on-free-frac 0.95 --on-stand 1.0 --diag-every 1 --diag-freeze"
  echo "$(date '+%F %T') dose: $CMD" | tee -a $S/stages.txt
  $CMD > $S/dose_train.log 2>&1 || die "dose run (log $S/dose_train.log)"
  $TR $PY dose --run $X --out $S/dose.json; rc=$?; cp $S/dose.json $RES/dose.json
  rm -rf $DATA_DIR/runs/op_parity/runs/$X; exit $rc ;;
pilot)
  AF=${2:?after ids}; ids=(); ar=()
  for A in ${ARMS_RUN:-$ARMS}; do T=${TAG[$A]}; [[ ! -e $DATA_DIR/runs/op_parity/runs/$T/ckpt-final.pt ]] || die "$T exists"
    t=$(PRIO=${TPR[$A]} sub stand-train-$T --after $AF --train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train --timeout-h 2 -- bash $ME train $A) || exit 1
    PR=$((${PRIO:-10} + 1))
    r=$(PRIO=$PR bash experiments/alpasim_proxy/scripts/wait_set.sh readc $T $t | tail -1); [[ -n $r ]] || die "readc $T"; echo "$r ws-collect-$T" >> $S/jobs.txt; ids+=($r)
    for i in 0 1 2 3 4 5; do ids+=($(PRIO=$PR TRIES=2 sub stand-held-$T-$i --after $t "${RD[@]}" -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1; done
    for c in $SUB; do ar+=($(PRIO=$PR TRIES=2 sub stand-ar-$T-c$c --after $t "${RD[@]}" -- bash $JT aread $T off $c)) || exit 1; done
    echo "$(date '+%F %T') pilot arm $A $T: train $t" | tee -a $S/stages.txt; done
  echo "${ARMS_RUN:-$ARMS}" > $S/arms.txt
  s=$(sub stand-sum --after $(csv ${ids[@]} ${ar[@]}) --vram 0.1 --cpu 32 --ram 96 --timeout-h 3 -- bash $ME sum) || exit 1
  echo "$(date '+%F %T') pilot: reads ${ids[*]} ${ar[*]} summary $s" | tee -a $S/stages.txt; echo "$s" ;;
train)   # train <arm>: JTNS1K-L18-s0's command + the arm's switches; refuses unless gate, identity and dose passed
  A=$2; T=${TAG[$A]}
  python3 - $RES <<'PY' || die "pre-checks failed: no training"
import json, sys
r = sys.argv[1]
g, i, d = (json.load(open(f"{r}/{n}.json")) for n in ("gate", "identity", "dose"))
assert g["gate"]["pass_"], "offline gate failed"
assert i["identical"], "switch-off identity failed"
assert d["in_range"], f"W_S {d['W_S']} out of range"
PY
  W=$(python3 -c "import json; print(json.load(open('$RES/dose.json'))['W_S'])")
  CMD="$(base $T 1000 50) $(term $A $W)"; echo "$(date '+%F %T') train $T: $CMD" | tee -a $S/stages.txt
  exec $CMD ;;
sum)     # after every read: extract (control + both arms), arena per arm, drift per arm, the rule per arm
  RUN=${ARMS_RUN:-$(cat $S/arms.txt)}
  $AP $FM ext --read $CTL=$J/arena/$CTL@off $(for A in $RUN; do echo ${TAG[$A]}=$J/arena/${TAG[$A]}@off; done) --out $S/ext || die "extract"
  runs=(); for t in $CTL:JTNS1K $(for A in $RUN; do echo ${TAG[$A]}:${LAB[$A]}; done); do
    d=$($TR -c "import sys; sys.path[:0] = ['experiments/defect_fix/scripts']; import df; print(','.join(sorted(df.read_dirs('${t%%:*}')['held'])))" | tail -1)
    [[ -n $d ]] || die "no held dirs ${t%%:*}"; runs+=("${t##*:}@18=$d"); done
  $AP experiments/route_follow/scripts/rf_follow.py --runs "${runs[@]}" --out $S/follow --cover none --workers 16 > $S/follow.log 2>&1 || die "rf_follow"
  for A in $RUN; do T=${TAG[$A]}
    $V $FM arena --src $S/ext --a $T --b $CTL --out $S/arena_$A.json || die "arena $A"
    $AP $PY drift --follow $S/follow --a ${LAB[$A]}@18 --b JTNS1K@18 --out $S/drift_$A.json || die "drift $A"
    cp $S/arena_$A.json $S/drift_$A.json $RES/
    $TR $PY pilot --tag $T --ctl $CTL --arena $S/arena_$A.json --drift $S/drift_$A.json --ext $S/ext --out $RES/pilot_$A.md | tee $S/pilot_$A.txt || die "summary $A"; done
  date '+%F %T' > $S/PILOT_DONE ;;
*) sed -n 2,9p $0; exit 2 ;;
esac
