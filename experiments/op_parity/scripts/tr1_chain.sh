#!/usr/bin/env bash
# op_parity lane TR1 (plans/2026-10-10-tr1-prereg.md): the adapter without the exported speed prior. One self-advancing chain per stage in
# tmux jev (scripts/tmux_run.sh tr1-<stage> experiments/op_parity/scripts/tr1_chain.sh <stage> [args]); every GPU job through the pool,
# navtest / navhard through jevdrive.bench, WOD val through the decision 155 harness (wod_slot.py report). Rerunning a stage resumes.
#   check        20-step parity of the modified pp_train.py (TR1 options off) against the unmodified one (git worktree of REF_SHA), then a
#                20-step smoke of every arm
#   s0           pilot scale (navsim/op-parity-s234, 3 000 steps x 64, 1 seed): lead-token pre-training -> the reference and the arms ->
#                navtest, the DIAG1 probes, WOD val -> tables and the stage gate
#   cl <name> <group=tag+tag ...>   AlpaSim nuPlan public 700 scenes (ot2_loop.py, OT_LANE=tr1) for the tags and P2H10-F at this checkout
#   s1 <arm...>  full scale (navsim/op-parity-full, 10 000 steps x 128, seeds 0 / 1) of the named arms: navtest, navhard, probes, WOD val
# State: $DATA_DIR/runs/op_parity/tr1/chain-<stage>/{STATUS, DONE, ERROR, log.txt, jobs.txt}; pool logs under .../tr1/pool/.
set -uo pipefail
cd "$(dirname "$0")/../../.."
STAGE=$1; shift
O=$DATA_DIR/runs/op_parity/tr1
D=$O/chain-$STAGE; mkdir -p "$D" "$O/tok"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
OP=$DATA_DIR/envs/openpilot/bin/python
J=$DATA_DIR/envs/jevdrive/bin/python
VPY=$PWD/.venv/bin/python
CL="$VPY -m jevdrive.cl"
S=experiments/op_parity/scripts
B=("$VPY" -m jevdrive.bench)
L=$O/pool
CK=$DATA_DIR/runs/op_parity/runs
ONNX=$DATA_DIR/runs/op_parity/hugsim/onnx
BIAS=$DATA_DIR/runs/op_parity/wod
RES=experiments/op_parity/results/tr1
REF_SHA=${REF_SHA:-9c592640}
H10="--hinge-lam 10 --hinge-margin 0.3"
PD="navtrain_full.s2of12 navtrain_full.s3of12 navtrain_full.s4of12"
FD=$(for i in $(seq 0 11); do echo -n "navtrain_full.s${i}of12 "; done)
PILOT="--frames warp --host --data $PD --split navsim/op-parity-s234 --steps 3000 --batch 64 --warmup 100 --eval-every 1000 $H10"
FULL="--frames warp --host --data $FD --split navsim/op-parity-full --steps 10000 --batch 128 --warmup 300 --eval-every 1000 $H10"
ARMS=(B0 NA NAH LD LG PO)
flags() {  # arm scale(P|F) -> pp_train flags of the arm
  local tok=$O/tok/$2.pt
  case $1 in
    B0)  echo "--arm P2";;
    NA)  echo "--arm P2 --ego-noax";;
    NAH) echo "--arm P2 --ego-noax --hist-cv";;
    LD)  echo "--arm P2L --ego-noax --hist-cv --lead-init $tok";;
    LG)  echo "--arm P2L --ego-noax --hist-cv --lead-init $tok --retime lead";;
    PO)  echo "--arm P2L --ego-noax --hist-cv --lead-init $tok --retime moving";;
    *) return 1;;
  esac
}
status() { echo "$(date '+%F %T') tr1 $STAGE: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return
        $CL queue 2>/dev/null | awk -v p="$ld/log.txt" '($2 == "queued" || $2 == "running" || $2 == "inbox") && index($0, p) {f = 1} END {exit !f}' && return
        rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity-tr1 --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n $ld" >> "$D/jobs.txt"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 20; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; return 0; }
serve() { local tag=$1                                   # WOD val, decision 155 harness (wod1_recipes_chain.sh)
          [[ -f $ONNX/pp-$tag.onnx ]] || $PY $S/pp_hugsim.py onnx --tag $tag --out $ONNX/pp-$tag.onnx >/dev/null || die "onnx $tag"
          [[ -f $BIAS/bias-$tag.npz ]] || $PY $S/pp_wod.py bias --tags $tag || die "bias $tag"
          sub tr1-wod $L/wod-$tag --vram 8 --cpu 14 --ram 40 -- $OP scripts/wod_zeroshot_openpilot.py --set rater extra --workers 12 \
              --onnx $ONNX/pp-$tag.onnx --tag $tag --bias $BIAS/bias-$tag.npz; }
TV="--train --vram 24 --cpu 4 --ram 40"

if [[ $STAGE == check ]]; then
  W=$O/ref-$REF_SHA
  [[ -d $W ]] || git worktree add --detach "$W" "$REF_SHA" || die "worktree"
  C20="--arm P2 --seed 0 ${PILOT/--steps 3000/--steps 20} --eval-every 20"
  C20=${C20/--eval-every 1000/}
  status "parity: unmodified ($REF_SHA) vs modified pp_train, 20 steps"
  sub tr1-par $L/par-old $TV -- $PY $W/$S/pp_train.py $C20 --tag tr1-par-old
  sub tr1-par $L/par-new $TV -- $PY $S/pp_train.py $C20 --tag tr1-par-new
  sub tr1-tok $L/tok-smoke --vram 6 --cpu 2 --ram 24 -- $PY $S/tr1.py tok --name smoke --data $PD --steps 200
  waitdirs $L/par-old $L/par-new $L/tok-smoke
  a=$(grep -h "step 20:\|dev @ 20" $L/par-old/log.txt | sed 's/.*\(step 20:\|dev @ 20:\)/\1/; s/; .*//'); b=$(grep -h "step 20:\|dev @ 20" $L/par-new/log.txt | sed 's/.*\(step 20:\|dev @ 20:\)/\1/; s/; .*//')
  echo "old: $a"; echo "new: $b"
  [[ -n $a && $a == "$b" ]] || die "parity: losses / dev differ"
  $PY - <<EOF || die "parity: checkpoints differ"
import torch
a, b = (torch.load("$CK/tr1-par-%s/ckpt-final.pt" % k, map_location="cpu", weights_only=False)["model"] for k in ("old", "new"))
d = max(float((a["net"][k].float() - b["net"][k].float()).abs().max()) for k in a["net"])
e = max(float((a["parity"][k].float() - b["parity"][k].float()).abs().max()) for k in a["parity"])
print("max |d| plan weights", d, "adapter", e, "tr1" in b)
assert d == 0 and e == 0 and "tr1" not in b
EOF
  status "20-step smokes of the arms"
  for arm in "${ARMS[@]:1}"; do f=$(flags $arm smoke)
    sub tr1-smoke $L/smoke-$arm $TV -- $PY $S/pp_train.py $f --seed 0 ${PILOT/--steps 3000/--steps 20} --tag tr1-smoke-$arm; done
  waitdirs $(for arm in "${ARMS[@]:1}"; do echo $L/smoke-$arm; done)
  grep -h "step 20:\|dev @ 20" $L/smoke-*/log.txt | sed 's/.*INFO *//' | cut -c1-260
  rm -rf $CK/tr1-par-old $CK/tr1-par-new; for arm in "${ARMS[@]:1}"; do rm -rf $CK/tr1-smoke-$arm; done
elif [[ $STAGE == s0 || $STAGE == s1 ]]; then
  if [[ $STAGE == s0 ]]; then SC=P; SEEDS="0"; RUNS=("${ARMS[@]}"); TA=$PILOT; TOKD="$PD"; TOKS=navsim/op-parity-s234
  else SC=F; SEEDS="0 1"; RUNS=("$@"); TA=$FULL; TOKD="$FD"; TOKS=navsim/op-parity-full; TV="--train --vram 36 --cpu 6 --ram 45"; fi
  REFT=$([[ $SC == P ]] && echo T1B0-P-s0 || echo P2H10-F-s0)
  status "lead-token pre-training ($SC)"
  sub tr1-tok $L/tok-$SC --vram 8 --cpu 2 --ram 40 -- $PY $S/tr1.py tok --name $SC --data $TOKD --split $TOKS
  waitdirs $L/tok-$SC; cat $O/tok/$SC.json
  status "trainings: ${RUNS[*]} x seeds $SEEDS"
  TAGS=""
  for arm in "${RUNS[@]}"; do f=$(flags $arm $SC) || die "unknown arm $arm"; for s in $SEEDS; do t=T1$arm-$SC-s$s; TAGS+="$t "
    sub tr1-t-$SC $L/t-$t $TV -- $PY $S/pp_train.py $f --seed $s $TA --tag $t; done; done
  waitdirs $(for t in $TAGS; do echo $L/t-$t; done)
  for t in $TAGS; do [[ -f $CK/$t/ckpt-final.pt ]] || die "no checkpoint $t"; done
  grep -h "dev @ [13]0*00:" $L/t-T1*-$SC-s*/log.txt | sed 's/.*INFO//' | cut -c1-300
  status "navtest (bench), probes, WOD val: $TAGS"
  "${B[@]}" run --model $TAGS --bench navtest || die "bench navtest"
  PT="$TAGS"; [[ $SC == F ]] && PT="P2H10-F-s0 P2H10-F-s1 $TAGS"
  sub tr1-probe $L/probe-$STAGE --vram 24 --cpu 8 --ram 48 -- $PY $S/tr1.py probe --name $STAGE --tags $PT
  if [[ $SC == F ]]; then G=$(for t in $TAGS; do echo -n "$t@gimm "; done); "${B[@]}" run --model $G --bench navhard || die "bench navhard"; fi
  for t in $TAGS; do serve $t; done
  "${B[@]}" status --model $TAGS --bench navtest --wait || die "navtest"
  [[ $SC == F ]] && { "${B[@]}" status --model $G --bench navhard --wait || die "navhard"; }
  waitdirs $L/probe-$STAGE $(for t in $TAGS; do echo $L/wod-$t; done)
  status "reports"
  mkdir -p $RES
  $VPY $S/tr1.py report --name $STAGE --ref $REFT || die "probe report"
  if [[ $SC == P ]]; then
    "${B[@]}" report --bench navtest --arms $(for a in "${RUNS[@]:1}"; do echo -n "$a=T1$a-P-s0 "; done) --vs B0=T1B0-P-s0 --out $RES/bench-s0 || die "navtest report"
    $J $S/wod_slot.py report --out $RES/wod-s0 --arms shipped P2H10=P2H10-F-s0+P2H10-F-s1 $(for a in "${RUNS[@]}"; do echo -n "$a=T1$a-P-s0 "; done) \
        --pairs $(for a in "${RUNS[@]:1}"; do echo -n "$a:B0 "; done) $(for a in "${RUNS[@]}"; do echo -n "$a:shipped "; done) || die "wod report"
  else
    AR=$(for a in "${RUNS[@]}"; do echo -n "$a=T1$a-F-s0+T1$a-F-s1 "; done)
    "${B[@]}" report --bench navtest --arms $AR --vs P2H10=P2H10-F-s0+P2H10-F-s1 --out $RES/bench-s1 || die "navtest report"
    "${B[@]}" report --bench navhard --arms $(for a in "${RUNS[@]}"; do echo -n "$a=T1$a-F-s0@gimm+T1$a-F-s1@gimm "; done) --vs P2H10=P2H10-F-s0@gimm+P2H10-F-s1@gimm --out $RES/bench-s1 || die "navhard report"
    $J $S/wod_slot.py report --out $RES/wod-s1 --arms shipped P2H10=P2H10-F-s0+P2H10-F-s1 WLG=WLG-full-s0+WLG-full-s1 $AR \
        --pairs $(for a in "${RUNS[@]}"; do echo -n "$a:P2H10 $a:shipped "; done) P2H10:shipped || die "wod report"
  fi
elif [[ $STAGE == cl ]]; then                              # cl <name> <group=tag+tag ...>: AlpaSim nuPlan public, the 700 scenes with OT3's chunk lists
  NAME=$1; shift
  RA=$DATA_DIR/runs/alpasim; AS=experiments/alpasim/scripts
  J=(); GR=()
  for t in P2H10-F-s0 P2H10-F-s1; do J+=("$t:sh30:SH30_TAG=$t:$t"); done            # the baseline at this checkout
  for g in "$@"; do GR+=(--group "$g"); for t in ${g#*=}; do :; done; IFS=+ read -ra TS <<< "${g#*=}"; for t in "${TS[@]}"; do J+=("$t:sh30:SH30_TAG=$t:$t"); done; done
  status "closed loop $NAME, code $(git rev-parse --short HEAD): ${J[*]}"
  OT_LANE=tr1 OT_PRIO=13 OT2_MAX_ACTIVE=${TR1_STACKS:-6} python3 $AS/ot2_loop.py $NAME "${J[@]}" || die "ot2_loop rc $? (see $RA/tr1/$NAME/ERROR)"
  mkdir -p $RES
  $VPY $AS/ot3_report.py --manifest $RA/tr1/$NAME/manifest.json --shards $RA/c0b/lists/shards.tsv --out $RA/tr1/results --name $NAME \
      --base P2H10=P2H10-F-s0+P2H10-F-s1 "${GR[@]}" > $D/report.log 2>&1 || die "report (see $D/report.log)"
  cp $RA/tr1/results/${NAME}_report.md $RES/alpasim_$NAME.md; cp $RA/tr1/results/${NAME}_stats.json $RES/alpasim_${NAME}_stats.json
else
  die "unknown stage $STAGE"
fi
status "done"; date > "$D/DONE"
