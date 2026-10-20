#!/usr/bin/env bash
# LON-ANCHOR (plans/2026-10-20-lon-anchor-prereg.md): the station-anchored two-sided arc term (lib/nr1_anchor.py, rm_train --on-anchor W),
# three doses + a same-step control, 1 000 steps from NTV0-L18-s0 on on1_JTCP (d402's merged pool) with the cp rows at half the S draw.
# Box, repo root; every job through the GPU pool, ids in $L/jobs.txt, stages in $L/stages.txt. Env PRIO (11).
#   lon_anchor.sh pre                labels -> gate (CPU): G1 + reported trigger statistics (supplement 1: labels v2, gate_v2)
#   lon_anchor.sh dry                the four training commands and their diffs against the control (only --tag, --on-anchor may differ)
#   lon_anchor.sh chain <old rev>    weights (needs experiments/pai_fix/scenes/) -> identity (the control's command 60 steps eager, <old rev>
#                                    vs this code) -> calibration (100 steps at W_PROBE, --diag-every 10) -> calib.json -> control + 3 arms ->
#                                    inertia.py cf / report on the 4 checkpoints (+ NTV0-L18-s0, RSG-L18-s0) -> the registered read-out
#   job bodies: labels | gate | weights | calrun | calib | train <arm> | cf | report
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
L=$DATA_DIR/runs/defect_fix/lon_anchor; ME=experiments/defect_fix/scripts/lon_anchor.sh; PY=experiments/defect_fix/scripts/lon_anchor.py
TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; V=.venv/bin/python; JT=experiments/defect_fix/scripts/jt.sh
LAT=$DATA_DIR/runs/defect_fix/lat/L18NR1v2; RES=experiments/defect_fix/results/lon_anchor; POOL=on1_JTCP; WF=LONP; INIT=NTV0-L18-s0; W_PROBE=0.004
IN=experiments/defect_fix/scripts/inertia.py; RUNS=$DATA_DIR/runs/op_parity/runs; GG=LONGG2; CAL=LONACAL2-L18-s0   # supplement 1 addendum: rerun tags
declare -A TAG=([C]=LONC1K-L18-s0 [H]=LONAH1K-L18-s0 [A]=LONA1K-L18-s0 [D]=LONAD1K-L18-s0) DOSE=([H]=A05 [A]=A10 [D]=A20)
FT=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train)
mkdir -p $L/pool $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $L/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner defect-fix --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-11} \
     --no-check --log-dir $L/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $L/jobs.txt; echo "$id"; }
base() {  # base <tag> <steps> [jt.sh]: NTV0-L18-s0's command (jt.sh train DRY, ONK=32, ARCROWS=brake) on on1_JTCP / LONP, warm-up 50, --init NTV0
  local C; C=$(DRY=1 ONK=32 ARCROWS=brake WARM=50 bash ${3:-$JT} train $1 $POOL $LAT $WF $2 0 | tail -1)
  [[ $C == *"--init RB0-F-s0 "* && $C == *"--on-k 32 "* && $C == *"--warmup 50 "* && $C == *"--steps $2 "* && $C == *"w/$WF.npz "* \
     && $C == *"--on-arc 0 "* && $C == *"--on $POOL "* ]] || die "unexpected command: $C"
  echo "${C/--init RB0-F-s0 /--init $INIT }"; }
dose() { python3 -c "import json; print(json.load(open('$L/calib.json'))['doses']['$1'])"; }
cmd() {  # cmd <arm>: the control's command; arms + --on-anchor <dose>
  local A=$1; [[ $A == C ]] && { base ${TAG[C]} 1000; return; }; echo "$(base ${TAG[$A]} 1000) --on-anchor $(dose ${DOSE[$A]})"; }
case ${1:-} in
pre)
  [[ ! -e $DATA_DIR/runs/op_parity/cache/$POOL/lon_anchor_v2.npz ]] || die "labels exist: never overwritten"
  git rev-parse --short HEAD > $L/code.txt
  l=$(sub lon-labels --vram 0.1 --cpu 32 --ram 120 --timeout-h 3 -- bash $ME labels) || exit 1
  g=$(sub lon-gate --after $l --vram 0.1 --cpu 8 --ram 64 --timeout-h 1 -- bash $ME gate) || exit 1
  echo "$(date '+%F %T') pre (code $(cat $L/code.txt)): labels $l gate $g" | tee -a $L/stages.txt ;;
dry)
  for A in C H A D; do echo "$A: $(cmd $A)"; done
  for A in H A D; do echo "== control -> $A"; diff <(cmd C | tr ' ' '\n') <(cmd $A | tr ' ' '\n'); done ;;
chain)
  rev=${2:?old rev}; WT=$DATA_DIR/tmp/df_old_$rev; [[ -d $WT ]] || git worktree add --detach $WT $rev > /dev/null || exit 1
  for X in $GG-old $GG-new $CAL ${TAG[@]}; do [[ ! -e $RUNS/$X/ckpt-final.pt ]] || die "$X exists"; done
  [[ -f $RES/gate_v2.json ]] && python3 -c "import json, sys; sys.exit(0 if json.load(open('$RES/gate_v2.json'))['G1']['pass_'] else 1)" || die "G1 not passed"
  ls experiments/pai_fix/scenes/* > /dev/null 2>&1 || die "experiments/pai_fix/scenes/ missing: the exam exclusion is required"
  h=$(git rev-parse --short HEAD); echo $h > $L/code_chain.txt
  w=$(sub lon-weights --vram 0.1 --cpu 2 --ram 24 --timeout-h 1 -- bash $ME weights) || exit 1
  FE=("${FT[@]}" --env JEV_RM_EAGER=1)
  a=$(sub lon-gg-old --after $w "${FE[@]}" --timeout-h 1 -- bash -c "$(base $GG-old 60 $WT/$JT)") || exit 1
  b=$(sub lon-gg-new --after $w "${FE[@]}" --timeout-h 1 -- bash -c "$(base $GG-new 60)") || exit 1
  e=$(sub lon-bitcmp --after $a,$b --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR experiments/defect_fix/scripts/df.py bitcmp --a $GG-old \
      --b $GG-new --out lon_anchor/identity --note "LON control command (NTV0 recipe on on1_JTCP / LONP) 60 steps eager: code $rev vs $h without --on-anchor") || exit 1
  c=$(sub lon-calrun --after $e "${FT[@]}" --timeout-h 1 -- bash $ME calrun) || exit 1
  k=$(sub lon-calib --after $c --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME calib) || exit 1
  t=(); for A in C H A D; do
    t+=($(sub lon-train-${TAG[$A]} --after $([[ $A == C ]] && echo $e || echo $k) "${FT[@]}" --timeout-h 2 -- bash $ME train $A)) || exit 1; done
  f=$(sub lon-cf --after $(IFS=,; echo "${t[*]}") --vram 24 --cpu 16 --ram 96 --timeout-h 2 -- bash $ME cf) || exit 1
  r=$(sub lon-report --after $f --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') chain (code $h, old $rev): weights $w identity $a $b bitcmp $e calrun $c calib $k train ${t[*]} cf $f report $r" | tee -a $L/stages.txt ;;
labels) $AP $PY labels --pool $POOL ;;
gate)   $TR $PY gate ;;
weights) $TR $PY weights --pool $POOL --base JT00S --share 0.5 --out $WF --require-fix ;;
calrun)  # the calibration: arm A's command at W_PROBE, 100 steps, --diag every 10 (its gradients set the dose; the checkpoint is removed)
  X=$CAL; CMD="$(base $X 100) --on-anchor $W_PROBE --diag-every 10"
  echo "$(date '+%F %T') calrun: $CMD" | tee -a $L/stages.txt
  $CMD > $L/calrun.log 2>&1 || die "calibration run (log $L/calrun.log)"; rm -rf $RUNS/$X ;;
calib) $TR $PY calib --tag $CAL --w $W_PROBE ;;
train)   # train <arm>: refuses unless G1, identity and (arms) the calibration are there
  A=$2; python3 - $RES $A <<'PY' || die "pre-checks failed: no training"
import json, sys
r, a = sys.argv[1], sys.argv[2]
assert json.load(open(f"{r}/gate_v2.json"))["G1"]["pass_"], "G1 (supplement 1) failed"
assert json.load(open(f"{r}/identity.json"))["identical"], "switch-off identity failed"
assert a == "C" or json.load(open(f"{r}/calib.json"))["w1"] > 0, "no calibration"
PY
  CMD="$(cmd $A)" || exit 1; echo "$(date '+%F %T') train ${TAG[$A]}: $CMD" | tee -a $L/stages.txt; exec $CMD ;;
cf)     $TR $IN cf --tags $INIT,${TAG[C]},${TAG[H]},${TAG[A]},${TAG[D]},RSG-L18-s0 --gate-tag RSG-L18-s0 --name lon_anchor ;;
report)
  $TR $IN report --name lon_anchor > $L/inertia_report.log 2>&1 || die "inertia report"
  cp experiments/defect_fix/results/inertia/report_lon_anchor.md $RES/inertia_report.md && cp experiments/defect_fix/results/inertia/report_lon_anchor.json $RES/inertia_report.json || die "no report"
  $TR $PY inertia --name lon_anchor --ctl ${TAG[C]} --arms ${TAG[H]} ${TAG[A]} ${TAG[D]} || die "read-out"; date '+%F %T' > $L/DONE ;;
*) sed -n 2,11p $0; exit 2 ;;
esac
