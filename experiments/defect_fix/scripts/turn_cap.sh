#!/usr/bin/env bash
# TURN-CAP (NR1 prereg supplement 21, plans/2026-10-20-nr1-supp21-turncap.md): a one-sided upper arc limit on moving S rows in sharp turns
# (lib/nr1_free.py turncap_*, rm_train --on-turncap W). Two 1 000-step pilots T1 / T3 from the stored B2 pilot STANDFMXH1K-L18-s0 (its stored
# reads are the control; it gets the turn-set reads here). Box, repo root; every job through the GPU pool, ids in $S/jobs.txt. Env PRIO (12;
# reads PRIO + 1).
#   turn_cap.sh check               DRY commands of T1 / T3 (W placeholders) against the stored B2 pilot command -> results/turn_cap/cmd_diff.txt
#   turn_cap.sh pre <old rev>       labels (CPU) -> offline gate (CPU); identity (B2 pilot command 60 steps eager, <old rev> vs this code);
#                                   dose (100 frozen eager steps with --on-turncap W0, --diag-every 1); the control's turn-set reads.
#                                   Prints the ids the pilot waits for
#   turn_cap.sh pilot <after ids>   T1, T3: train (checks gate / identity / dose first) -> wait set v2, held 460 @18, arena subset (6 chunks),
#                                   turn-set reads -> psum -> $S/STAGE1_DONE
#   turn_cap.sh pai <arm>           stage 2 (only for an arm that passed stage 1 on the Mac): its PAI half -> paigate -> $S/PAI_<arm>_DONE
#   job bodies: label | gate | dose | train <arm> | tread <tag> <shard> | psum | paigate <arm>
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/defect_fix; S=$O/turn_cap; J=$O/jt; ME=experiments/defect_fix/scripts/turn_cap.sh; PY=experiments/defect_fix/scripts/turn_cap.py
SP=experiments/defect_fix/scripts/stand.py; FM=experiments/defect_fix/scripts/free_m.py; JT=experiments/defect_fix/scripts/jt.sh
TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; V=.venv/bin/python
RES=experiments/defect_fix/results/turn_cap; CTL=STANDFMXH1K-L18-s0; CL=STANDFMXH; SUB="0 3 6 9 12 15"
B2CMD=$O/stand_b/cmd_B2.txt; REF=$DATA_DIR/runs/route_follow/acc_gap/read/s2; PCH="c000 c002 c004 c006"; CTLPAI=$O/stand_r/pai/$CTL
declare -A TAG=([T1]=TCAPA1K-L18-s0 [T3]=TCAPB1K-L18-s0) LAB=([T1]=TCAPA [T3]=TCAPB)
PAISERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10
AH=$DATA_DIR/runs/alpasim_proxy/arena/hold; AC=$DATA_DIR/datasets/alpasim_nuplan_arena/v1; KEY=runtime.scene_provider.trajdata
RD=(--vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4)
TRN=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train)
mkdir -p $S/pool $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $S/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner defect-fix --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-12} \
     --no-check --log-dir $S/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $S/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
shards() { ls $S/lists/ct*.txt 2>/dev/null | sed 's/.*ct\([0-9]*\)\.txt/\1/' | sort -n | paste -sd' '; }
cmd() {  # cmd <tag> <W> [steps]: the stored B2 pilot command, tag swapped, + --on-turncap W; [steps] for the identity / dose runs
  local T=$1 W=$2 C; C=$(cat $B2CMD) || die "no $B2CMD"
  [[ $C == *" --tag $CTL "* && $C == *" --steps 1000 "* && $C != *"--on-turncap"* ]] || die "unexpected B2 command"
  C=${C/ --tag $CTL / --tag $T }; [[ -n ${3:-} ]] && C=${C/ --steps 1000 / --steps $3 }
  echo "$C --on-turncap $W"; }
dose_w() { python3 -c "import json; print(json.load(open('$RES/dose.json'))['W']['$1'])"; }
treads() {  # treads <tag> [after]: the turn-set reads of one tag (one job per cache shard) -> ids
  local T=$1 ids=() k; for k in $(shards); do
    ids+=($(PRIO=$((${PRIO:-12} + 1)) TRIES=2 sub tc-tr-$T-ct$k ${2:+--after $2} "${RD[@]}" -- bash $ME tread $T $k)) || return 1; done; csv ${ids[@]}; }
pai() {  # pai <tag> <out> [after]: PAI half (c000 / c002 / c004 / c006) with the d366 serve -> prints the job ids
  CHUNKS_FROM=$S/chunks4 SERVE="$PAISERVE" WIZ=$WIZ CONC=4 PRIO=$((${PRIO:-12} + 1)) AFTER=${3:-} bash experiments/alpasim/scripts/pai_box.sh submit $2 $1 \
    > $S/pai_$1.submit.txt || die "pai submit $1"; awk '$2 ~ /^pai-/' $S/pai_$1.submit.txt >> $S/jobs.txt; awk '$2 ~ /^pai-/ {print $1}' $S/pai_$1.submit.txt | paste -sd,; }
case ${1:-} in
check)
  cp $B2CMD $S/cmd_ctl.txt; for A in T1 T3; do cmd ${TAG[$A]} W_$A > $S/cmd_$A.txt || exit 1; done
  { for A in T1 T3; do echo "== $CTL -> ${TAG[$A]}"; diff <(tr ' ' '\n' < $S/cmd_ctl.txt) <(tr ' ' '\n' < $S/cmd_$A.txt); done; } > $RES/cmd_diff.txt
  cat $RES/cmd_diff.txt
  for A in T1 T3; do got=$(diff <(tr ' ' '\n' < $S/cmd_ctl.txt) <(tr ' ' '\n' < $S/cmd_$A.txt) | grep '^>' | cut -c3- | paste -sd' ')
    [[ $got == "${TAG[$A]} --on-turncap W_$A" ]] || die "arm $A: new tokens '$got'"; done
  echo "diff ok" ;;
pre)
  rev=${2:?old rev}; WT=$DATA_DIR/tmp/df_old_$rev; [[ -d $WT ]] || git worktree add --detach $WT $rev > /dev/null || exit 1
  for X in TCGG-old TCGG-new TCDOSE; do [[ ! -e $DATA_DIR/runs/op_parity/runs/$X/ckpt-final.pt ]] || die "$X exists"; done
  [[ ! -e $DATA_DIR/runs/op_parity/cache/on1_JTX/turn_cap.npz ]] || die "turn_cap.npz exists: never overwritten"
  mkdir -p $S/chunks4; for c in $PCH; do cp $REF/chunks/$c.txt $S/chunks4/; done
  $V $PY lists > $S/shards.txt || die "lists"; [[ -n $(shards) ]] || die "no turn-set lists"
  h=$(git rev-parse --short HEAD); B=$(sed "s/ --tag $CTL / --tag TCGG-old /; s/ --steps 1000 / --steps 60 /" $B2CMD)
  l=$(sub tc-label --vram 0.1 --cpu 8 --ram 64 --timeout-h 1 -- $TR $PY label --pool on1_JTX) || exit 1
  g=$(sub tc-gate --after $l --vram 0.1 --cpu 18 --ram 96 --timeout-h 3 -- $TR $PY gate --out $S/gate --workers 18) || exit 1
  FE=("${TRN[@]}" --env JEV_RM_EAGER=1 --timeout-h 1)
  a=$(sub tc-gg-old "${FE[@]}" -- bash -c "cd $WT && $B") || exit 1
  b=$(sub tc-gg-new "${FE[@]}" -- bash -c "${B/TCGG-old/TCGG-new}") || exit 1
  e=$(sub tc-bitcmp --after $a,$b --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR experiments/defect_fix/scripts/df.py bitcmp --a TCGG-old \
      --b TCGG-new --out turn_cap/identity --note "B2 pilot command (STAND 0.0068 + FREE-M 0.001) 60 steps eager: code $rev vs $h without --on-turncap") || exit 1
  d=$(sub tc-dose --after $l "${FE[@]}" -- bash $ME dose) || exit 1
  q=$(treads $CTL) || die "control turn-set reads"
  echo "$(date '+%F %T') pre (code $h): label $l gate $g identity $a $b bitcmp $e dose $d control turn-set reads $q" | tee -a $S/stages.txt
  echo "$g,$e,$d" ;;
label) $TR $PY label --pool on1_JTX ;;
gate) $TR $PY gate --out $S/gate ;;
dose)    # 100 frozen eager steps, both FREE-M (B2's 0.001) and TURN-CAP (W0 0.004) on, per-step trunk gradients -> dose.json (W of T1 / T3)
  X=TCDOSE; CMD="$(cmd $X 0.004 100) --diag-every 1 --diag-freeze"; echo "$(date '+%F %T') dose: $CMD" | tee -a $S/stages.txt
  $CMD > $S/dose_train.log 2>&1 || die "dose run (log $S/dose_train.log)"
  $TR $PY dose --run $X --log $S/dose_train.log --out $RES/dose.json; rc=$?; rm -rf $DATA_DIR/runs/op_parity/runs/$X; exit $rc ;;
pilot)
  AF=${2:?after ids}; ids=(); ar=(); PR=$((${PRIO:-12} + 1))
  for A in T1 T3; do T=${TAG[$A]}; [[ ! -e $DATA_DIR/runs/op_parity/runs/$T/ckpt-final.pt ]] || die "$T exists"
    t=$(sub tc-train-$T --after $AF "${TRN[@]}" --timeout-h 2 -- bash $ME train $A) || exit 1
    r=$(PRIO=$PR bash experiments/alpasim_proxy/scripts/wait_set.sh readc $T $t | tail -1); [[ -n $r ]] || die "readc $T"; echo "$r ws-collect-$T" >> $S/jobs.txt; ids+=($r)
    for i in 0 1 2 3 4 5; do ids+=($(PRIO=$PR TRIES=2 sub tc-held-$T-$i --after $t "${RD[@]}" -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1; done
    for c in $SUB; do ar+=($(PRIO=$PR TRIES=2 sub tc-ar-$T-c$c --after $t "${RD[@]}" -- bash $JT aread $T off $c)) || exit 1; done
    q=$(treads $T $t) || die "turn-set reads $T"; ar+=($q)
    echo "$(date '+%F %T') pilot arm $A $T: train $t" | tee -a $S/stages.txt; done
  cq=$(awk '$2 ~ /^tc-tr-'$CTL'-/ {print $1}' $S/jobs.txt | paste -sd,)
  p=$(sub tc-psum --after $(csv ${ids[@]} ${ar[@]}),$cq --vram 0.1 --cpu 32 --ram 96 --timeout-h 3 -- bash $ME psum) || exit 1
  echo "$(date '+%F %T') pilot: reads ${ids[*]} ${ar[*]} control turn-set $cq psum $p" | tee -a $S/stages.txt; echo "psum $p" ;;
train)
  A=$2; python3 - $RES <<'PY' || die "pre-checks failed: no training"
import json, sys
r = sys.argv[1]
g, i, d = (json.load(open(f"{r}/{n}.json")) for n in ("gate", "identity", "dose"))
assert g["pass_"], "offline gate failed"
assert i["identical"], "switch-off identity failed"
assert d["pass_"], "dose / smoke failed"
PY
  CMD="$(cmd ${TAG[$A]} $(dose_w $A))"; echo "$(date '+%F %T') train ${TAG[$A]}: $CMD" | tee -a $S/stages.txt; exec $CMD ;;
tread)   # tread <tag> <shard>: the turn-set windows of one cache shard outside the subset chunks (= jt.sh aread on $S/lists/ct<shard>.txt)
  T=$2; k=$3; D=$S/arena/$T/ct$k/run
  [[ -f $D/aggregate/results-summary.json ]] && { echo "$D already read"; exit 0; }
  source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
  rm -rf $D; mkdir -p $D; rc=0
  env -u JEV_HOLD SH30_TAG=$T bash experiments/alpasim/scripts/run.sh $D sh30hold --scene-list $S/lists/ct$k.txt "${BASE[@]}" "$KEY.cache_location=$AC/s$k" || rc=$?
  if [[ $rc -ne 0 ]]; then
    grep -q "exit code -9" "$D/native_summary.json" 2>/dev/null && { echo "runtime worker killed (-9): exit 137 for the pool's kill retry" >&2; exit 137; }
    die "run.sh rc $rc ($D)"; fi
  python3 - "$D" <<'PY'
import glob, json, os, sys
D = sys.argv[1]; n = 0
for r in json.load(open(f"{D}/aggregate/results-summary.json"))["rollouts"]:
    if not r.get("failure_reason") and (r.get("score") or 0) > 0:
        for f in glob.glob(f"{D}/rollouts/{r['clipgt_id']}/*/rollout.asl"):
            os.remove(f); n += 1
print("pruned", n, "rollout.asl")
PY
  ;;
psum)    # both arms against the control: read roots (subset chunks + turn-set shards), extract, arena, drift, stand.py's table (its verdict is not ours)
  X=""; for t in $CTL ${TAG[T1]} ${TAG[T3]}; do R=$S/reads/$t; rm -rf $R; mkdir -p $R
    for c in $SUB; do [[ -d $J/arena/$t@off/c$c/run ]] || die "no $t c$c"; ln -s $J/arena/$t@off/c$c $R/c$c; done
    for k in $(shards); do [[ -d $S/arena/$t/ct$k/run ]] || die "no $t ct$k"; ln -s $S/arena/$t/ct$k $R/ct$k; done; X="$X $t=$R"; done
  $AP $FM ext --read $X --out $S/ext || die "extract"
  runs=(); for t in $CTL:$CL ${TAG[T1]}:${LAB[T1]} ${TAG[T3]}:${LAB[T3]}; do
    d=$($TR -c "import sys; sys.path[:0] = ['experiments/defect_fix/scripts']; import df; print(','.join(sorted(df.read_dirs('${t%%:*}')['held'])))" | tail -1)
    [[ -n $d ]] || die "no held dirs ${t%%:*}"; runs+=("${t##*:}@18=$d"); done
  $AP experiments/route_follow/scripts/rf_follow.py --runs "${runs[@]}" --out $S/follow --cover none --workers 16 > $S/follow.log 2>&1 || die "rf_follow"
  for A in T1 T3; do T=${TAG[$A]}
    $V $FM arena --src $S/ext --a $T --b $CTL --out $S/arena_$A.json || die "arena $A"
    $AP $SP drift --follow $S/follow --a ${LAB[$A]}@18 --b $CL@18 --out $S/drift_$A.json || die "drift $A"
    cp $S/arena_$A.json $S/drift_$A.json $RES/
    $TR $SP pilot --tag $T --ctl $CTL --arena $S/arena_$A.json --drift $S/drift_$A.json --ext $S/ext --out $RES/s17table_$A.md > $S/s17_$A.txt || die "table $A"; done
  $V -c "import pandas as pd; w = pd.read_parquet('$S/ext/win.parquet'); w[['tag', 'id', 'score', 'type']].to_csv('$RES/arena_windows.csv.gz', index=False)" || die "windows"
  date '+%F %T' > $S/STAGE1_DONE ;;
pai)     # stage 2 for one arm (after its stage 1 passed on the Mac); the control's PAI half is STAND-R's (read-only, same chunks and serve)
  A=${2:?arm}; T=${TAG[$A]}; mkdir -p $S/chunks4; for c in $PCH; do cp $REF/chunks/$c.txt $S/chunks4/; done
  q=$(pai $T $S/pai/$T); [[ -n $q ]] || die "PAI $T"; cq=$(cat $O/stand_r/ctl_pai_ids.txt)
  g=$(sub tc-paigate-$A --after $q,$cq --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME paigate $A) || exit 1
  echo "$(date '+%F %T') stage 2 $A: PAI $q gate $g" | tee -a $S/stages.txt; echo "paigate $g" ;;
paigate)
  A=$2; T=${TAG[$A]}
  $V experiments/practice_pai/scripts/pp.py gate --pilot $S/pai/$T --control $CTLPAI --ref $REF --chunks $(echo $PCH | tr ' ' ,) --out $S/pai_gate_$A
  [[ -f $S/pai_gate_$A/gate.json ]] || die "PAI gate $A"; cp $S/pai_gate_$A/gate.json $RES/pai_half_$A.json; cp $S/pai_gate_$A/gate.md $RES/pai_half_$A.md
  date '+%F %T' > $S/PAI_${A}_DONE ;;
*) sed -n 2,15p $0; exit 2 ;;
esac
