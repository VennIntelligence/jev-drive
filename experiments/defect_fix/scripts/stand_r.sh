#!/usr/bin/env bash
# STAND-R (NR1 prereg supplement 20, plans/2026-10-20-nr1-supp20-standr.md): STAND's rows extended to a rolling ego (V_STAND <= v0 < V_ROLL)
# while the human stands all 4 s, cap aligned to the human's standing station (lib/nr1_free.py, rm_train --on-stand-roll V). Two 1 000-step
# pilots R1 / R2 from the stored B2 pilot STANDFMXH1K-L18-s0 (its stored reads are the control). Box, repo root; every job through the GPU
# pool, ids in $S/jobs.txt. Env PRIO (12; reads PRIO + 1).
#   stand_r.sh s3read              diagnostic (main session 10-20): B2 s1 (STANDFMXH-L18-s1) with JEV_HOLD=stop3 on the 18 arena held chunks
#                                  -> s3sum (extract with the native read, windows table -> results/stand_r/)
#   stand_r.sh check               DRY commands of R1 / R2 against the stored B2 pilot command -> results/stand_r/cmd_diff.txt
#   stand_r.sh pre <old rev>       gap labels (CPU) -> offline gate (CPU); identity (B2 pilot command 60 steps eager, <old rev> vs this code);
#                                  smoke (R2's command 60 steps eager); the control's PAI half. Prints the ids the pilot waits for
#   stand_r.sh pilot <after ids>   R1, R2: train (checks gate / identity / smoke first) -> wait set v2, held 460 @18, arena subset (6 chunks)
#                                  -> psum (extract, arena, drift, stand.py pilot table per arm) -> $S/STAGE1_DONE
#   stand_r.sh pai <arm>           stage 2 (only for an arm that passed stage 1 on the Mac): its PAI half -> paigate -> $S/PAI_<arm>_DONE
#   job bodies: gap | gate | smoke | train <arm> | psum | paigate <arm> | s3sum
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/defect_fix; S=$O/stand_r; J=$O/jt; ME=experiments/defect_fix/scripts/stand_r.sh; PY=experiments/defect_fix/scripts/stand_r.py
SP=experiments/defect_fix/scripts/stand.py; FM=experiments/defect_fix/scripts/free_m.py; JT=experiments/defect_fix/scripts/jt.sh
TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; V=.venv/bin/python
RES=experiments/defect_fix/results/stand_r; CTL=STANDFMXH1K-L18-s0; CL=STANDFMXH; SUB="0 3 6 9 12 15"; VR=3.0
B2CMD=$O/stand_b/cmd_B2.txt; F=$O/free_m; REF=$DATA_DIR/runs/route_follow/acc_gap/read/s2; PCH="c000 c002 c004 c006"
declare -A TAG=([R1]=STANDRH1K-L18-s0 [R2]=STANDR2H1K-L18-s0) LAB=([R1]=STANDRH [R2]=STANDR2H) WS=([R1]=0.0068 [R2]=0.0136)
PAISERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10
AH=$DATA_DIR/runs/alpasim_proxy/arena/hold; RD=(--vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4)
TRN=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train)
B2S1=STANDFMXH-L18-s1
mkdir -p $S/pool $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $S/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner defect-fix --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-12} \
     --no-check --log-dir $S/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $S/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
cmd() {  # cmd <arm> [tag] [steps]: the stored B2 pilot command, tag swapped, + --on-stand-roll VR (R2: W_S doubled); [steps] for the smoke
  local A=$1 T=${2:-${TAG[$1]}} C; C=$(cat $B2CMD) || die "no $B2CMD"
  [[ $C == *" --tag $CTL "* && $C == *" --on-stand 0.0068 "* && $C == *" --steps 1000 "* ]] || die "unexpected B2 command"
  C=${C/ --tag $CTL / --tag $T }; C=${C/ --on-stand 0.0068 / --on-stand ${WS[$A]} }; [[ -n ${3:-} ]] && C=${C/ --steps 1000 / --steps $3 }
  echo "$C --on-stand-roll $VR"; }
pai() {  # pai <tag> <out> [after]: PAI half (c000 / c002 / c004 / c006) with the d366 serve -> prints the job ids
  CHUNKS_FROM=$S/chunks4 SERVE="$PAISERVE" WIZ=$WIZ CONC=4 PRIO=$((${PRIO:-12} + 1)) AFTER=${3:-} bash experiments/alpasim/scripts/pai_box.sh submit $2 $1 \
    > $S/pai_$1.submit.txt || die "pai submit $1"; awk '$2 ~ /^pai-/' $S/pai_$1.submit.txt >> $S/jobs.txt; awk '$2 ~ /^pai-/ {print $1}' $S/pai_$1.submit.txt | paste -sd,; }
case ${1:-} in
s3read)
  PR=$((${PRIO:-12} + 1)); nch=$(python3 -c "import json; print(len(json.load(open('$AH/meta.json'))['chunks']))"); fr=()
  for c in $(seq 0 $((nch - 1))); do fr+=($(PRIO=$PR TRIES=2 sub str-ar-$B2S1-s3-c$c "${RD[@]}" -- bash $JT aread $B2S1 stop3 $c)) || exit 1; done
  f=$(sub str-s3sum --after $(csv ${fr[@]}) --vram 0.1 --cpu 32 --ram 96 --timeout-h 3 -- bash $ME s3sum) || exit 1
  echo "$(date '+%F %T') s3read: reads ${fr[*]} s3sum $f" | tee -a $S/stages.txt; echo "s3sum $f" ;;
s3sum)
  $AP $FM ext --read "$B2S1+stop3=$J/arena/$B2S1@stop3" "$B2S1=$J/arena/$B2S1@off" --out $S/ext_s3 || die "extract stop3"
  $V -c "import pandas as pd; w = pd.read_parquet('$S/ext_s3/win.parquet'); w[['tag', 'id', 'score', 'type']].to_csv('$RES/arena_windows_H_s1_stop3.csv.gz', index=False)" || die "windows"
  date '+%F %T' > $S/S3_DONE ;;
check)
  mkdir -p $S; cp $B2CMD $S/cmd_ctl.txt; for A in R1 R2; do cmd $A > $S/cmd_$A.txt || exit 1; done
  { for A in R1 R2; do echo "== $CTL -> ${TAG[$A]}"; diff <(tr ' ' '\n' < $S/cmd_ctl.txt) <(tr ' ' '\n' < $S/cmd_$A.txt); done; } > $RES/cmd_diff.txt
  cat $RES/cmd_diff.txt
  for A in R1 R2; do got=$(diff <(tr ' ' '\n' < $S/cmd_ctl.txt) <(tr ' ' '\n' < $S/cmd_$A.txt) | grep '^>' | cut -c3- | paste -sd' ')
    want="${TAG[$A]} $([[ $A == R2 ]] && echo '0.0136 ')--on-stand-roll $VR"; [[ $got == "$want" ]] || die "arm $A: new tokens '$got', want '$want'"; done
  echo "diff ok" ;;
pre)
  rev=${2:?old rev}; WT=$DATA_DIR/tmp/df_old_$rev; [[ -d $WT ]] || git worktree add --detach $WT $rev > /dev/null || exit 1
  for X in STRGG-old STRGG-new STRGG-R2; do [[ ! -e $DATA_DIR/runs/op_parity/runs/$X/ckpt-final.pt ]] || die "$X exists"; done
  [[ ! -e $DATA_DIR/runs/op_parity/cache/on1_JTX/stand_gap.npz ]] || die "stand_gap.npz exists: never overwritten"
  mkdir -p $S/chunks4; for c in $PCH; do cp $REF/chunks/$c.txt $S/chunks4/; done
  h=$(git rev-parse --short HEAD); B=$(sed "s/ --tag $CTL / --tag STRGG-old /; s/ --steps 1000 / --steps 60 /" $B2CMD)
  l=$(sub str-gap --vram 0.1 --cpu 32 --ram 96 --timeout-h 2 -- $TR $PY gap --pool on1_JTX --workers 32) || exit 1
  g=$(sub str-gate --after $l --vram 0.1 --cpu 16 --ram 64 --timeout-h 2 -- $TR $PY gate --out $S/gate) || exit 1
  FE=("${TRN[@]}" --env JEV_RM_EAGER=1 --timeout-h 1)
  a=$(sub str-gg-old "${FE[@]}" -- bash -c "cd $WT && $B") || exit 1
  b=$(sub str-gg-new "${FE[@]}" -- bash -c "${B/STRGG-old/STRGG-new}") || exit 1
  e=$(sub str-bitcmp --after $a,$b --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR experiments/defect_fix/scripts/df.py bitcmp --a STRGG-old \
      --b STRGG-new --out stand_r/identity --note "B2 pilot command (STAND 0.0068 + FREE-M 0.001) 60 steps eager: code $rev vs $h without --on-stand-roll") || exit 1
  k=$(sub str-smoke --after $l "${FE[@]}" -- bash $ME smoke) || exit 1
  q=$(pai $CTL $S/pai/$CTL); [[ -n $q ]] || die "control PAI"
  echo "$q" > $S/ctl_pai_ids.txt
  echo "$(date '+%F %T') pre (code $h): gap $l gate $g identity $a $b bitcmp $e smoke $k control PAI $q" | tee -a $S/stages.txt
  echo "$g,$e,$k" ;;
gap) $TR $PY gap --pool on1_JTX --workers 32 ;;
gate) $TR $PY gate --out $S/gate ;;
smoke)   # R2's command, 60 steps eager: finite losses, the rolling part fires at some step
  X=STRGG-R2; CMD="$(cmd R2 $X 60)"; echo "$(date '+%F %T') smoke: $CMD" | tee -a $S/stages.txt
  $CMD > $S/smoke_train.log 2>&1 || die "smoke run (log $S/smoke_train.log)"
  $TR $PY smoke --run $X --log $S/smoke_train.log --out $RES/smoke.json; rc=$?; rm -rf $DATA_DIR/runs/op_parity/runs/$X; exit $rc ;;
pilot)
  AF=${2:?after ids}; ids=(); ar=(); PR=$((${PRIO:-12} + 1))
  for A in R1 R2; do T=${TAG[$A]}; [[ ! -e $DATA_DIR/runs/op_parity/runs/$T/ckpt-final.pt ]] || die "$T exists"
    t=$(sub str-train-$T --after $AF "${TRN[@]}" --timeout-h 2 -- bash $ME train $A) || exit 1
    r=$(PRIO=$PR bash experiments/alpasim_proxy/scripts/wait_set.sh readc $T $t | tail -1); [[ -n $r ]] || die "readc $T"; echo "$r ws-collect-$T" >> $S/jobs.txt; ids+=($r)
    for i in 0 1 2 3 4 5; do ids+=($(PRIO=$PR TRIES=2 sub str-held-$T-$i --after $t "${RD[@]}" -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1; done
    for c in $SUB; do ar+=($(PRIO=$PR TRIES=2 sub str-ar-$T-c$c --after $t "${RD[@]}" -- bash $JT aread $T off $c)) || exit 1; done
    echo "$(date '+%F %T') pilot arm $A $T: train $t" | tee -a $S/stages.txt; done
  p=$(sub str-psum --after $(csv ${ids[@]} ${ar[@]}) --vram 0.1 --cpu 32 --ram 96 --timeout-h 3 -- bash $ME psum) || exit 1
  echo "$(date '+%F %T') pilot: reads ${ids[*]} ${ar[*]} psum $p" | tee -a $S/stages.txt; echo "psum $p" ;;
train)
  A=$2; python3 - $RES <<'PY' || die "pre-checks failed: no training"
import json, sys
r = sys.argv[1]
g, i, s = (json.load(open(f"{r}/{n}.json")) for n in ("gate", "identity", "smoke"))
assert g["pass_"], "offline gate failed"
assert i["identical"], "switch-off identity failed"
assert s["pass_"], "smoke failed"
PY
  CMD="$(cmd $A)"; echo "$(date '+%F %T') train ${TAG[$A]}: $CMD" | tee -a $S/stages.txt; exec $CMD ;;
psum)    # both arms against the control (stored reads): extract, arena, drift, stand.py's table (its verdict is supplement 17's, not ours)
  X="$CTL=$J/arena/$CTL@off ${TAG[R1]}=$J/arena/${TAG[R1]}@off ${TAG[R2]}=$J/arena/${TAG[R2]}@off"
  $AP $FM ext --read $X --out $S/ext || die "extract"
  runs=(); for t in $CTL:$CL ${TAG[R1]}:${LAB[R1]} ${TAG[R2]}:${LAB[R2]}; do
    d=$($TR -c "import sys; sys.path[:0] = ['experiments/defect_fix/scripts']; import df; print(','.join(sorted(df.read_dirs('${t%%:*}')['held'])))" | tail -1)
    [[ -n $d ]] || die "no held dirs ${t%%:*}"; runs+=("${t##*:}@18=$d"); done
  $AP experiments/route_follow/scripts/rf_follow.py --runs "${runs[@]}" --out $S/follow --cover none --workers 16 > $S/follow.log 2>&1 || die "rf_follow"
  for A in R1 R2; do T=${TAG[$A]}
    $V $FM arena --src $S/ext --a $T --b $CTL --out $S/arena_$A.json || die "arena $A"
    $AP $SP drift --follow $S/follow --a ${LAB[$A]}@18 --b $CL@18 --out $S/drift_$A.json || die "drift $A"
    cp $S/arena_$A.json $S/drift_$A.json $RES/
    $TR $SP pilot --tag $T --ctl $CTL --arena $S/arena_$A.json --drift $S/drift_$A.json --ext $S/ext --drift-guard --out $RES/s17table_$A.md > $S/s17_$A.txt || die "table $A"; done
  $V -c "import pandas as pd; w = pd.read_parquet('$S/ext/win.parquet'); w[['tag', 'id', 'score', 'type']].to_csv('$RES/arena_windows.csv.gz', index=False)" || die "windows"
  date '+%F %T' > $S/STAGE1_DONE ;;
pai)     # stage 2 for one arm (after its stage 1 passed on the Mac)
  A=${2:?arm}; T=${TAG[$A]}; q=$(pai $T $S/pai/$T); [[ -n $q ]] || die "PAI $T"; cq=$(cat $S/ctl_pai_ids.txt)
  g=$(sub str-paigate-$A --after $q,$cq --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME paigate $A) || exit 1
  echo "$(date '+%F %T') stage 2 $A: PAI $q gate $g" | tee -a $S/stages.txt; echo "paigate $g" ;;
paigate)
  A=$2; T=${TAG[$A]}
  $V experiments/practice_pai/scripts/pp.py gate --pilot $S/pai/$T --control $S/pai/$CTL --ref $REF --chunks $(echo $PCH | tr ' ' ,) --out $S/pai_gate_$A
  [[ -f $S/pai_gate_$A/gate.json ]] || die "PAI gate $A"; cp $S/pai_gate_$A/gate.json $RES/pai_half_$A.json; cp $S/pai_gate_$A/gate.md $RES/pai_half_$A.md
  date '+%F %T' > $S/PAI_${A}_DONE ;;
*) sed -n 2,15p $0; exit 2 ;;
esac
