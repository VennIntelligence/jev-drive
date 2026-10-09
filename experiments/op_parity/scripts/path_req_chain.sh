#!/usr/bin/env bash
# op_parity path-req (plans/2026-10-09-path-req-prereg.md, decision 204): one self-advancing chain in tmux jev
# (scripts/tmux_run.sh path-req experiments/op_parity/scripts/path_req_chain.sh). Every GPU / CPU job through the pool, every navtest read through
# jevdrive.bench. Arms = the SH30 pilot recipe of decisions 197 / 200 (P2, hinge lambda 30 / margin 0.5 m, navsim/op-parity-s234, 3 000 steps x 64)
# + pp_train --mem-e2e q<kind> (scripts/path_req.py): degraded fields of the LOGGED FUTURE trajectory as adapter memory.
# LEAKED-LABEL ORACLE PROBES, never a reportable driver (QH reads a thin head's prediction instead: a reference point, not a method).
# Baseline GH0-F-s{0,1} is decision 197's checkpoint (reused, not retrained).
#   stage 0  reliability: QF (tokenizer pre-trained with its own head) / QFC (random init) / QFL (random init, tokenizer lr 1e-3) x 3 seeds, QX x 2
#            -> dev gate (QF reads on 3 / 3 seeds) -> navtest QF, QF masked, QX -> navtest gate
#   stage 1  cross-track error axis (4 levels x 2 seeds; trained while stage 0 is scored, scored only after the navtest gate)
#   stage 2  along-track error, horizon, content, and the thin-head path QH
# State: $DATA_DIR/runs/op_parity/path_req/chain/{STATUS, DONE, ERROR, log.txt, jobs.txt}. Rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/op_parity/path_req
D=$O/chain; mkdir -p "$D" "$O/tok"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
NAV=$DATA_DIR/envs/navsim2/bin/python
VPY=$PWD/.venv/bin/python
CL="$DATA_DIR/envs/jevdrive/bin/python -m jevdrive.cl"
S=experiments/op_parity/scripts
B=("$VPY" -m jevdrive.bench)
L=$D/pool
MEM=$DATA_DIR/runs/op_parity/mem
DATA="navtrain_full.s2of12 navtrain_full.s3of12 navtrain_full.s4of12"
SPLIT=navsim/op-parity-s234
HF="--hinge-lam 30 --hinge-margin 0.5"
VRAM=${PR_VRAM:-24}          # GB. Measured: the 30-step, 64-row smoke incl. dev eval + navtest export peaks at 18.5 (pool record); decision 200's
                             # path-field arm peaked at 18.2 over 3 000 steps
declare -A KIND=([QF]=qf [QFC]=qf [QFL]=qf [QX]=qx [QN025]=qn0.25 [QN050]=qn0.5 [QN100]=qn1.0 [QN200]=qn2.0 [QL075]=ql0.75 [QL150]=ql1.5 [QL300]=ql3.0
                 [QT1]=qt1 [QT2]=qt2 [QS]=qs [QV]=qv [QCH]=qch [QC7]=qc7 [QH]=qh)
ST0=(QF QFC QFL QX); ST1=(QN025 QN050 QN100 QN200); ST2=(QL075 QL150 QL300 QT1 QT2 QS QV QCH QC7 QH)
seeds() { case $1 in QF|QFC|QFL) echo 0 1 2;; *) echo 0 1;; esac; }
status() { echo "$(date '+%F %T') op_parity path-req: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '($2 == "queued" || $2 == "running") && ($3 == n || $4 == n) {print $1; exit}')   # a queued row has no card column
        [[ -n $live ]] && return; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 20; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; return 0; }
declare -A TOKD
tok() {  # kind: the tokenizer of the kind, pre-trained with its own thin head (one job per kind)
  [[ -n ${TOKD[$1]:-} || -f $O/tok/$1.pt ]] && return; TOKD[$1]=1
  sub pr-tok-$1 $L/tok-$1 --vram 3 --cpu 2 --ram 16 -- $PY $S/path_req.py tok --kind $1
}
train() {  # arm seed
  local arm=$1 t=$1-F-s$2 k=${KIND[$1]} extra=() gate=()
  case $arm in
    QFC) ;;
    QFL) extra=(--mem-lr 1e-3);;
    QX|QH) extra=(--mem-init $O/tok/qf.pt); gate=(--when-exists $O/tok/qf.pt);;
    *) tok $k; extra=(--mem-init $O/tok/$k.pt); gate=(--when-exists $O/tok/$k.pt);;
  esac
  [[ $arm == QH ]] && gate=(--when-exists $O/head/head.json)
  sub pr-t-$t $L/t-$t --train --vram $VRAM --cpu 4 --ram 40 "${gate[@]}" -- $PY $S/pp_train.py --arm P2 --mem-e2e $k "${extra[@]}" --seed $2 \
      --frames warp --host --data $DATA --split $SPLIT --steps 3000 --batch 64 --warmup 100 --eval-every 1000 $HF --tag $t
}
launch() { for arm in "$@"; do for s in $(seeds $arm); do train $arm $s; done; done; }
settle() {  # arms: wait for their trainings, sanity-check them
  for arm in "$@"; do for s in $(seeds $arm); do
    local t=$arm-F-s$s
    waitdirs $L/t-$t
    $PY $S/pp_full_check.py train --tag $t || die "training sanity $t"
    [[ -f $MEM/ge_$t/lb_navtest.npy ]] || die "no navtest bank for $t"
  done; done
}
score() {  # replay-name spec ...: navtest through bench, then the four_dirs replay of the unmasked specs
  local name=$1; shift
  status "navtest ($name): $*"
  "${B[@]}" run --model "$@" --bench navtest || die "bench navtest"
  "${B[@]}" status --model "$@" --bench navtest --wait || die "navtest"
  local un=(); for x in "$@"; do [[ $x == *:noside ]] || un+=($x); done
  sub pr-replay-$name $L/replay-$name --vram 0.5 --cpu 36 --ram 64 -- $NAV $S/turn_oracle.py replay --name $name --models "${un[@]}"
  waitdirs $L/replay-$name
}
specs() {  # arms -> every seed, plus the memory-masked read of seed 0 (QF: of every seed)
  local out=()
  for arm in "$@"; do for s in $(seeds $arm); do
    out+=($arm-F-s$s)
    [[ $arm != QX && ( $s == 0 || $arm == QF ) ]] && out+=($arm-F-s$s:noside)
  done; done
  echo "${out[@]}"
}
report() { rm -rf $L/report; sub pr-report $L/report --vram 0.5 --cpu 8 --ram 32 -- $PY $S/path_req.py report --replays geo_s0 geo_e2e "$@"; waitdirs $L/report; }
drop_banks() { for arm in "$@"; do for s in $(seeds $arm); do rm -rf "$MEM/ge_$arm-F-s$s"; done; done; }   # this lane's own navtest banks, after scoring

for t in GH0-F-s0 GH0-F-s1; do [[ -f $DATA_DIR/runs/op_parity/runs/$t/ckpt-final.pt ]] || die "missing reused checkpoint $t"; done
for d in $DATA lb_navtest; do [[ -f $MEM/geo_x/$d.perm.npy ]] || die "missing geo_x permutation of $d"; done

status "stage 0: reliability (QF QFC QFL x 3 seeds, QX x 2)"
tok qf
launch "${ST0[@]}"
settle "${ST0[@]}"
$PY $S/path_req.py gate --stage dev || die "reliability gate (dev): the undegraded oracle is not read on 3 / 3 seeds"
status "stage 0: dev gate met; stage 1 trainings submitted, stage 0 navtest"
launch "${ST1[@]}"
score pr0 $(specs QF QX)
$PY $S/path_req.py gate --stage navtest; rc=$?
[[ $rc == 3 ]] && { report pr0; status "STOP: the undegraded oracle is itself below the +0.5 line"; drop_banks "${ST0[@]}"; date > "$D/DONE"; exit 0; }
[[ $rc == 0 ]] || die "reliability gate (navtest) not met"
drop_banks "${ST0[@]}"

status "stage 1: cross-track error axis"
settle "${ST1[@]}"
score pr1 $(specs "${ST1[@]}")
report pr0 pr1
cp -r $O/report $O/report-stage1
drop_banks "${ST1[@]}"

status "stage 2: thin head, along-track error, horizon, content"
sub pr-head $L/head --vram 12 --cpu 4 --ram 48 -- $PY $S/path_req.py head
launch "${ST2[@]}"
settle "${ST2[@]}"
score pr2 $(specs "${ST2[@]}")
report pr0 pr1 pr2
drop_banks "${ST2[@]}"
status "done"; date > "$D/DONE"
