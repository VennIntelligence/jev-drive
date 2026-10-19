#!/usr/bin/env bash
# NR1STOPPT (NR1 prereg supplement 18): "do not pass the human's stop point" on approach states (lib/nr1_free.py stoppt_*, rm_train --on-stoppt W).
# Box, repo root; every job through the GPU pool, ids in $S/jobs.txt. Env PRIO (10; reads PRIO + 1).
#   stoppt.sh check                the DRY commands of both arms against JTNS1K-L18-s0's -> $S/cmd_*.txt, results/stoppt/cmd_diff.txt
#   stoppt.sh pre <old rev>        labels -> gate (CPU); merge on1_JTPAI -> weights JTPAI00 / labels; identity (60 steps eager, <old rev> vs
#                                  this code); smoke (arm B's command, 100 steps); probe: A0 rerun (8 chunks, rollout.asl + dumps) -> rows ->
#                                  cache -> repro / clean -> features -> probe. Prints the ids the pilot waits for (gate, bitcmp, smoke)
#   stoppt.sh pilot <after ids>    STOPPT1K-L18-s0 (A), STOPPTP1K-L18-s0 (B): train (checks gate / identity / smoke first), wait set v2, held 460
#                                  @18, arena subset (6 chunks); per arm a psub job after its training submits its PAI half, extract, arena,
#                                  PAI gate and the arm's summary
#   stoppt.sh pilotc               (supplement 18 addendum) arm C STOPPTF1K-L18-s0 = arm B + the focused draw JTPAIF: weights -> train -> the same
#                                  reads -> psub C -> sum C (+ held 460 drift against the control, overrun zeros by scene); PRIO 12
#   job bodies: gate | smoke | train <arm> | rows | cache | clean | tax <tag> <read dir> | psub <arm> | sum <arm>
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/defect_fix; S=$O/stoppt; J=$O/jt; F=$O/free_m; ME=experiments/defect_fix/scripts/stoppt.sh; PY=experiments/defect_fix/scripts/stoppt.py
FM=experiments/defect_fix/scripts/free_m.py; TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; V=.venv/bin/python
JT=experiments/defect_fix/scripts/jt.sh; LAT=$O/lat/L18NR1v2; RES=experiments/defect_fix/results/stoppt; CTL=JTNS1K-L18-s0; SUB="0 3 6 9 12 15"; W=${W:-0.004}
CR=$DATA_DIR/runs/op_parity/cache; AS=$DATA_DIR/third_party/alpasim; RF=experiments/route_follow/scripts
declare -A TAG=([A]=STOPPT1K-L18-s0 [B]=STOPPTP1K-L18-s0 [C]=STOPPTF1K-L18-s0) POOL=([A]=on1_JT [B]=on1_JTPAI [C]=on1_JTPAI) WF=([A]=JT00 [B]=JTPAI00 [C]=JTPAIF)
PARTS="on1_L18RB0:rb0:self on1_L18RTB1:rtb1:self on1_L18YS:ys:self on1_L18Y:y:self on1_NR1TV:tv:both"   # = jt.sh, on1_JT's parts in order
A0=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0; A0R=$S/a0r; STG=$DATA_DIR/tmp/stoppt_a0r
REF=$DATA_DIR/runs/route_follow/acc_gap/read/s2; PCH="c000 c002 c004 c006"
PAISERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10
RD=(--vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4); FT=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train)
mkdir -p $S/pool $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $S/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner defect-fix --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-10} \
     --no-check --log-dir $S/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $S/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
base() {  # base <tag> <pool> <wfile> <steps> <warm>: JTN's command (jt.sh train DRY, ARCROWS=brake), checked
  local C; C=$(DRY=1 ARCROWS=brake WARM=$5 bash $JT train $1 $2 $LAT $3 $4 0 | tail -1)
  [[ $C == *"--on-k 70 "* && $C == *"w/$3.npz "* && $C == *"--on $2 "* && $C == *"--warmup $5 "* && $C == *"--tag $1 "* && $C == *"--steps $4 "* \
     && $C == *"--init RB0-F-s0 "* ]] || die "unexpected command: $C"; echo "$C"; }
arm() { echo "$(base ${TAG[$1]} ${POOL[$1]} ${WF[$1]} 1000 50) --on-stoppt $W"; }
pai() {  # pai <tag> <out> [after]: PAI half with the d366 serve -> prints the job ids
  CHUNKS_FROM=$S/chunks4 SERVE="$PAISERVE" WIZ=$WIZ CONC=4 PRIO=$((${PRIO:-10} + 1)) AFTER=${3:-} bash experiments/alpasim/scripts/pai_box.sh submit $2 $1 \
    > $S/pai_$1.submit.txt || die "pai submit $1"; awk '$2 ~ /^pai-/' $S/pai_$1.submit.txt >> $S/jobs.txt; awk '$2 ~ /^pai-/ {print $1}' $S/pai_$1.submit.txt | paste -sd,; }
case ${1:-} in
check)
  base $CTL on1_JT JT00 1000 50 > $S/cmd_ctl.txt || exit 1
  for A in A B; do arm $A > $S/cmd_$A.txt || exit 1; done
  { for A in A B; do echo "== $CTL -> ${TAG[$A]}"; diff <(tr ' ' '\n' < $S/cmd_ctl.txt) <(tr ' ' '\n' < $S/cmd_$A.txt); done; } > $RES/cmd_diff.txt
  cat $RES/cmd_diff.txt
  for A in A B; do n=$(diff <(tr ' ' '\n' < $S/cmd_ctl.txt) <(tr ' ' '\n' < $S/cmd_$A.txt) | grep -c '^>'); want=$([[ $A == A ]] && echo 3 || echo 5)
    [[ $n == "$want" ]] || die "arm $A: $n new tokens, want $want"; done; echo "diff ok" ;;
pre)
  rev=${2:?old rev}; WT=$DATA_DIR/tmp/df_old_$rev; [[ -d $WT ]] || git worktree add --detach $WT $rev > /dev/null || exit 1
  for X in STOPPTGG-old STOPPTGG-new STOPPTSM-L18-s0; do [[ ! -e $DATA_DIR/runs/op_parity/runs/$X/ckpt-final.pt ]] || die "$X exists"; done
  for x in $CR/on1_JTPAI $A0R/roll; do [[ ! -e $x ]] || die "$x exists: never overwritten"; done
  git rev-parse --short HEAD > $S/code.txt
  l=$(sub stoppt-labels --vram 0.1 --cpu 32 --ram 120 --timeout-h 3 -- $AP $PY labels --workers 32) || exit 1
  g=$(sub stoppt-gate --after $l --vram 0.1 --cpu 24 --ram 96 --timeout-h 3 -- bash $ME gate) || exit 1
  m=$(sub stoppt-merge --vram 0.1 --cpu 4 --ram 60 --timeout-h 3 -- $TR $RF/rf_pait.py merge --pools $PARTS on1_PPAI160:pai:self --out JTPAI --nan-keys fut_reg) || exit 1
  w=$(sub stoppt-weights --after $m --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- $TR $PY weights) || exit 1
  s=$(sub stoppt-assemble --after $l,$m --vram 0.1 --cpu 2 --ram 32 --timeout-h 1 -- $AP $PY assemble --pool on1_JTPAI) || exit 1
  FE=("${FT[@]}" --env JEV_RM_EAGER=1 --env ARCROWS=brake); h=$(git rev-parse --short HEAD)
  a=$(sub stoppt-gg-old "${FE[@]}" --timeout-h 1 -- bash $WT/$JT train STOPPTGG-old on1_JT $LAT JT00 60 0) || exit 1
  b=$(sub stoppt-gg-new "${FE[@]}" --timeout-h 1 -- bash $JT train STOPPTGG-new on1_JT $LAT JT00 60 0) || exit 1
  e=$(sub stoppt-bitcmp --after $a,$b --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR experiments/defect_fix/scripts/df.py bitcmp --a STOPPTGG-old \
      --b STOPPTGG-new --out stoppt/identity --note "JTN's command (JT00, arc 0, arc rows brake) 60 steps eager: code $rev vs $h without --on-stoppt") || exit 1
  k=$(sub stoppt-smoke --after $w,$s "${FT[@]}" --timeout-h 1 -- bash $ME smoke) || exit 1
  mkdir -p $A0R
  CHUNKS_FROM=$A0/chunks SERVE="$PAISERVE PAI_DUMP=100000" WIZ=$WIZ KEEP_ASL=1 PRIO=$((${PRIO:-10} + 1)) bash experiments/alpasim/scripts/pai_box.sh submit \
    $A0R/roll NTV0-L18-s0 > $A0R/roll_submit.txt || die "A0 rerun submit"
  [[ $(wc -l < $A0R/roll/jobs.txt) == 8 ]] || die "A0 rerun: not 8 chunks"; cat $A0R/roll/jobs.txt >> $S/jobs.txt; q=$(awk '{print $1}' $A0R/roll/jobs.txt | paste -sd,)
  r=$(sub stoppt-a0r-rows --after $q --vram 0.5 --cpu 24 --ram 96 --timeout-h 4 -- bash $ME rows) || exit 1
  c=$(sub stoppt-a0r-cache --after $r --vram 20 --cpu 8 --ram 64 --timeout-h 3 -- bash $ME cache) || exit 1
  x=$(sub stoppt-a0r-clean --after $c --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME clean) || exit 1
  f=$(sub stoppt-pfeats --after $c,$l --vram 20 --cpu 8 --ram 96 --timeout-h 2 -- $TR $PY pfeats) || exit 1
  p=$(sub stoppt-probe --after $f --vram 16 --cpu 16 --ram 128 --timeout-h 3 -- bash -c "$TR $PY probe && $TR $PY report > $RES/probe.md") || exit 1
  echo "$(date '+%F %T') pre (code $h): labels $l gate $g merge $m weights $w assemble $s identity $a $b bitcmp $e smoke $k a0r $q rows $r cache $c" \
       "clean $x pfeats $f probe $p" | tee -a $S/stages.txt
  echo "$g,$e,$k" ;;
gate)
  $AP $PY gate --out $S/gate; rc=$?; exit $rc ;;
smoke)   # arm B's command (merged pool, both row sources), 100 steps, --diag every step
  X=STOPPTSM-L18-s0; CMD="$(base $X on1_JTPAI JTPAI00 100 5) --on-stoppt $W --diag-every 10"
  echo "$(date '+%F %T') smoke: $CMD" | tee -a $S/stages.txt
  $CMD > $S/smoke_train.log 2>&1 || die "smoke run (log $S/smoke_train.log)"
  $TR $PY smoke --run $X --out $RES/smoke.json; rc=$?; rm -rf $DATA_DIR/runs/op_parity/runs/$X; exit $rc ;;
rows)
  mkdir -p $STG $A0R/rows
  env PAI_DUMP=100000 JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD $AS/.venv/bin/python experiments/onpol_pai/scripts/onp_rows.py rows --pass A0R \
      --runs "$A0R/roll/runs/*" --out $A0R/rows --stage $STG --tag NTV0-L18-s0 --jobs 24 || exit 1
  env PAI_DUMP=100000 JEV_REPO=$PWD PYTHONPATH=$PWD/experiments/alpasim/lib:$PWD $TR experiments/onpol_pai/scripts/onp_rows.py cold --pass A0R --out $A0R/rows --stage $STG --jobs 24 || exit 1
  env PYTHONPATH=$PWD/experiments/alpasim/lib:$PWD $TR experiments/onpol_pai/scripts/onp_rows.py report --pass A0R --out $A0R/rows > $A0R/rows/report_A0R.log 2>&1
  tail -3 $A0R/rows/report_A0R.log; rm -rf $STG ;;
cache) $TR experiments/onpol_pai/scripts/onp_cache.py build --pass A0R --rows $A0R/rows/A0R --out $A0R/cache ;;
clean)   # the repro read-out first (needs only the summaries), then the rollout.asl files and frame dumps (the rows and cache are built)
  $V $PY repro || true
  n=$(find $A0R/roll/runs -name rollout.asl | wc -l); find $A0R/roll/runs -name rollout.asl -delete; rm -rf $A0R/roll/runs/*/sim/driver-logs/dump
  echo "removed $n rollout.asl and the frame dumps; $(du -sh $A0R | cut -f1) left" ;;
pilotc)
  PRIO=${PRIO:-12}; export PRIO; T=${TAG[C]}; [[ ! -e $DATA_DIR/runs/op_parity/runs/$T/ckpt-final.pt ]] || die "$T exists"
  [[ -f $S/tax/$CTL.pkl ]] || die "control extract missing"
  wj=$(sub stoppt-weightsf --vram 0.1 --cpu 4 --ram 48 --timeout-h 1 -- $AP $PY weightsf) || exit 1
  t=$(sub stoppt-train-$T --after $wj "${FT[@]}" --timeout-h 2 -- bash $ME train C) || exit 1
  PR=$((PRIO + 1)); ids=()
  r=$(PRIO=$PR bash experiments/alpasim_proxy/scripts/wait_set.sh readc $T $t | tail -1); [[ -n $r ]] || die "readc $T"; echo "$r ws-collect-$T" >> $S/jobs.txt; ids+=($r)
  for i in 0 1 2 3 4 5; do ids+=($(PRIO=$PR TRIES=2 sub stoppt-held-$T-$i --after $t "${RD[@]}" -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1; done
  ar=(); for c in $SUB; do ar+=($(PRIO=$PR TRIES=2 sub stoppt-ar-$T-c$c --after $t "${RD[@]}" -- bash $JT aread $T off $c)) || exit 1; done
  printf '%s\n' "${ids[*]} ${ar[*]}" "$(csv ${ar[@]})" "$wj" > $S/pilot_C_ids.txt
  p=$(sub stoppt-psub-$T --after $t --env PRIO=$PRIO --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $ME psub C) || exit 1
  echo "$(date '+%F %T') pilot arm C $T: weights $wj train $t wait set $r held/arena ${ids[*]:1} ${ar[*]} psub $p" | tee -a $S/stages.txt ;;
pilot)
  AF=${2:?after ids}; ids=(); mkdir -p $S/chunks4; for c in $PCH; do cp $REF/chunks/$c.txt $S/chunks4/; done
  [[ -e $DATA_DIR/runs/op_parity/runs/$CTL/ckpt-final.pt ]] || die "$CTL missing"
  tc=$(sub stoppt-tax-$CTL --vram 0.1 --cpu 16 --ram 40 --timeout-h 1 -- bash $ME tax $CTL $F/pai/$CTL) || exit 1
  for A in A B; do T=${TAG[$A]}; [[ ! -e $DATA_DIR/runs/op_parity/runs/$T/ckpt-final.pt ]] || die "$T exists"
    t=$(sub stoppt-train-$T --after $AF "${FT[@]}" --timeout-h 2 -- bash $ME train $A) || exit 1
    PR=$((${PRIO:-10} + 1)); ids=()
    r=$(PRIO=$PR bash experiments/alpasim_proxy/scripts/wait_set.sh readc $T $t | tail -1); [[ -n $r ]] || die "readc $T"; echo "$r ws-collect-$T" >> $S/jobs.txt; ids+=($r)
    for i in 0 1 2 3 4 5; do ids+=($(PRIO=$PR TRIES=2 sub stoppt-held-$T-$i --after $t "${RD[@]}" -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1; done
    ar=(); for c in $SUB; do ar+=($(PRIO=$PR TRIES=2 sub stoppt-ar-$T-c$c --after $t "${RD[@]}" -- bash $JT aread $T off $c)) || exit 1; done
    printf '%s\n' "${ids[*]} ${ar[*]}" "$(csv ${ar[@]})" "$tc" > $S/pilot_${A}_ids.txt
    p=$(sub stoppt-psub-$T --after $t --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $ME psub $A) || exit 1
    echo "$(date '+%F %T') pilot arm $A $T: train $t wait set $r held/arena ${ids[*]:1} ${ar[*]} psub $p (control tax $tc)" | tee -a $S/stages.txt; done ;;
train)   # train <arm>: JTNS1K-L18-s0's command (+ arm B's pool and draw) + --on-stoppt W; refuses unless gate, identity and smoke passed
  A=$2; T=${TAG[$A]}
  python3 - $RES <<'PY' || die "pre-checks failed: no training"
import json, sys
r = sys.argv[1]
g, i, s = (json.load(open(f"{r}/{n}.json")) for n in ("gate", "identity", "smoke"))
assert g["gate"]["pass_"], "offline gate failed"
assert i["identical"], "switch-off identity failed"
assert s["pass_"], "smoke failed"
PY
  [[ $A != C ]] || [[ -f $RES/w_JTPAIF.json && -f $O/w/JTPAIF.npz ]] || die "arm C: no JTPAIF weights"
  CMD="$(arm $A)"; echo "$(date '+%F %T') train $T: $CMD" | tee -a $S/stages.txt
  exec $CMD ;;
tax)     # tax <tag> <PAI read dir>: rf_tax extract of its rollouts (the overrun count)
  mkdir -p $S/tax; $AP $RF/rf_tax.py extract --runs $(ls -d $3/runs/*/sim) --out $S/tax/$2.pkl --workers 16 > $S/tax/$2.log 2>&1 || die "extract $2" ;;
psub)    # psub <arm>: after the arm's training: its PAI half, then extract / arena / PAI gate / overrun extract / summary behind every read
  A=$2; T=${TAG[$A]}; { read -r ALL; read -r ARC; read -r TC; } < $S/pilot_${A}_ids.txt
  pp=$(pai $T $S/pai/$T); [[ -n $pp ]] || die "PAI $T"
  tx=$(sub stoppt-tax-$T --after $pp --vram 0.1 --cpu 16 --ram 40 --timeout-h 1 -- bash $ME tax $T $S/pai/$T) || exit 1
  e=$(sub stoppt-ext-$A --after $ARC --vram 0.1 --cpu 32 --ram 96 --timeout-h 2 -- $AP $FM ext --read $CTL=$J/arena/$CTL@off $T=$J/arena/$T@off --out $S/ext_$A) || exit 1
  x=$(sub stoppt-arena-$A --after $e --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- $V $FM arena --src $S/ext_$A --a $T --b $CTL --out $S/arena_$A.json) || exit 1
  g=$(sub stoppt-paigate-$A --after $pp --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash -c "$V experiments/practice_pai/scripts/pp.py gate --pilot $S/pai/$T \
      --control $F/pai/$CTL --ref $REF --chunks $(echo $PCH | tr ' ' ,) --out $S/pai_gate_$A; [[ -f $S/pai_gate_$A/gate.json ]]") || exit 1   # its own rule: not ours
  s=$(sub stoppt-sum-$A --after $(csv $ALL),$x,$g,$tx,$TC --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME sum $A) || exit 1
  echo "$(date '+%F %T') psub $A: PAI $pp tax $tx extract $e arena $x PAI gate $g summary $s" | tee -a $S/stages.txt ;;
sum)
  A=$2; T=${TAG[$A]}
  $AP $PY overrun --tax $CTL=$S/tax/$CTL.pkl $T=$S/tax/$T.pkl --out $S/overrun_$A.json || die "overrun $A"
  cp $S/arena_$A.json $S/overrun_$A.json $RES/; cp $S/pai_gate_$A/gate.md $RES/pai_half_$A.md
  DR=(); if [[ $A == C ]]; then         # held 460 right drift against the control (d377), as stand.sh sum
    runs=(); for t in $CTL:JTNS1K $T:STOPPTF; do
      d=$($TR -c "import sys; sys.path[:0] = ['experiments/defect_fix/scripts']; import df; print(','.join(sorted(df.read_dirs('${t%%:*}')['held'])))" | tail -1)
      [[ -n $d ]] || die "no held dirs ${t%%:*}"; runs+=("${t##*:}@18=$d"); done
    $AP experiments/route_follow/scripts/rf_follow.py --runs "${runs[@]}" --out $S/follow_C --cover none --workers 16 > $S/follow_C.log 2>&1 || die "rf_follow"
    $AP experiments/defect_fix/scripts/stand.py drift --follow $S/follow_C --a STOPPTF@18 --b JTNS1K@18 --out $S/drift_C.json || die "drift C"
    cp $S/drift_C.json $RES/; DR=(--drift $S/drift_C.json); fi
  $TR $PY pilot --tag $T --ctl $CTL --arena $S/arena_$A.json --pai $S/pai_gate_$A/gate.json --overrun $S/overrun_$A.json --ext $S/ext_$A \
      "${DR[@]}" --pai-dirs $F/pai/$CTL,$S/pai/$T --out $RES/pilot_$A.md | tee $S/pilot_$A.txt || die "summary $A"
  date '+%F %T' > $S/PILOT_${A}_DONE ;;
*) sed -n 2,13p $0; exit 2 ;;
esac
