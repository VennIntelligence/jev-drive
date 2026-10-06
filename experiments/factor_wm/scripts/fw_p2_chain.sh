#!/usr/bin/env bash
# factor_wm P2 on-policy lane (plans/2026-10-06-p2-onpolicy-prereg.md): one-shot chain, run in tmux jev via scripts/tmux_run.sh.
# Every GPU job goes through the pool; this driver only submits, waits and checks. Staged launch (docs/long-runs.md):
#   1   prep + preflights (collect --limit 2, train 5 steps)
#   ~10 seed 0: round-1 collection -> check -> rest of the DAgger chain (X) and the off-policy control (C) -> gate reads
#       (navtest W, navhard G, HUGSIM 12 x exam / spec, engine) -> gate (exit 3 = stop, the lane ends with the gate as its result)
#   all seed 1 chain, then every readout for both seeds -> report
# State: $DATA_DIR/runs/factor_wm/p2op/{STATUS, DONE, ERROR, STOPPED, log.txt, jobs.txt}. Rerunning resumes (finished jobs are skipped).
set -uo pipefail
cd "$(dirname "$0")/../../.."
D=$DATA_DIR/runs/factor_wm/p2op; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
JEV=$DATA_DIR/envs/jevdrive/bin/python
NAV2=$DATA_DIR/envs/navsim2/bin/python
HPY=$DATA_DIR/envs/hugsim/bin/python
CL="$JEV -m jevdrive.cl"
S=experiments/factor_wm/scripts
PS=experiments/op_parity/scripts
L=$D/pool
H12=$S/p2op_hug12.txt
H64=experiments/hugsim/scripts/derot_all64.txt
STEPS=(0 800 1600 2400)
status() { echo "$(date '+%F %T') fw p2op: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; kill -TERM $$ 2>/dev/null; exit 1; }   # $$ = the driver, also from a $(...) subshell
# sub NAME LOGDIR args... -> job id ("done" when LOGDIR/DONE exists; the live id when queued / running under the same name)
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && { echo done; return; }
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && { echo "$live"; return; }; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner "factor_wm p2op" --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; echo "$id"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 60; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; }
aft() { local ids; ids=$(printf '%s\n' "$@" | tr ',' '\n' | grep -v -e '^done$' -e '^$' | paste -sd, -); [[ -n $ids ]] && echo "--after $ids"; }

PF_COL="$PY $S/fw_p2.py collect --policy P2-F-s0 --tag pf-xr --seed 1 --limit 2"
PF_TRAIN="$PY $S/fw_p2.py train --arm C --init P2-F-s0 --steps 5 --limit 4096 --tag pf-C"
collect() {  # seed round policy after... -> comma-joined ids
  local s=$1 r=$2 pol=$3; shift 3; local ids=() pf=()
  for i in 0 1 2; do
    pf=(); [[ $s == 0 && $r == 1 && $i == 0 ]] && pf=(--preflight "$PF_COL")
    ids+=("$(sub fwp-col-xr$r-s$s-$i $L/col-xr$r-s$s-$i --vram 16 --cpu 18 --ram 40 "${pf[@]}" $(aft "$@") -- \
      $PY $S/fw_p2.py collect --policy $pol --tag xr$r-s$s --seed $r --shard $i/3)")
  done; local IFS=,; echo "${ids[*]}"; }
train() {  # tag arm seed rolls steps after...
  local tag=$1 arm=$2 s=$3 rolls=$4 steps=$5; shift 5; local pf=()
  [[ $tag == PC-s0 ]] && pf=(--preflight "$PF_TRAIN")
  sub fwp-t-$tag $L/t-$tag --train --vram 40 --cpu 8 --ram 80 "${pf[@]}" $(aft "$@") -- \
    $PY $S/fw_p2.py train --arm $arm --init P2-F-s$s --rolls $rolls --steps $steps --seed $s --tag $tag; }
dagger_rest() {  # seed after-collect-round-1 -> id of the final X training
  local s=$1 c1=$2
  local t1 c2 t2 c3
  t1=$(train PX-r1-s$s X $s xr1-s$s ${STEPS[1]} "$c1")
  c2=$(collect $s 2 PX-r1-s$s "$t1")
  t2=$(train PX-r2-s$s X $s xr1-s$s,xr2-s$s ${STEPS[2]} "$c2")
  c3=$(collect $s 3 PX-r2-s$s "$t2")
  train PX-s$s X $s xr1-s$s,xr2-s$s,xr3-s$s ${STEPS[3]} "$c3"; }
evalj() { sub fwp-ev-$1 $L/ev-$1 --vram 16 --cpu 18 --ram 40 $(aft "${@:2}") -- $PY $S/fw_p2.py eval --policy $1 --tag ev-$1 --sets g0b,g1s; }
onnx() { sub fwp-onnx-$1 $L/onnx-$1 --vram 6 --cpu 4 --ram 16 $(aft "${@:2}") -- \
  $PY $PS/pp_hugsim.py onnx --tag $1 --out $DATA_DIR/runs/op_parity/hugsim/onnx/pp-$1.onnx; }
hug() {  # preset tag list name after...
  local pr=$1 t=$2 list=$3 nm=$4; shift 4
  sub fwp-h$nm-$pr-$t $L/h$nm-$pr-$t --vram 40 --cpu 14 --ram 45 --env PRESET=$pr $(aft "$@") -- bash $PS/pp_hugsim.sh arm $t $list 6; }
nt_plans() {  # frames name models... (after: env AFTER)
  local F=$1 nm=$2; shift 2
  sub fwp-ntp-$nm $L/ntp-$nm --vram 30 --cpu 8 --ram 40 $(aft $AFTER) -- $PY $PS/pp_eval.py --data lb_navtest --frames $F plans --models "$@" --tag p2op; }
nt_score() { local F=$1 nm=$2; shift 2
  sub fwp-nts-$nm $L/nts-$nm --vram 1 --cpu 24 --ram 48 --env NAVSIM_THREADS=22 $(aft $AFTER) -- $PY $PS/pp_eval.py --data lb_navtest --frames $F score --models "$@"; }
nh() {  # frames name models... -> id of the last harness job (after: env AFTER)
  local F=$1 nm=$2; shift 2; local p e ids=()
  p=$(sub fwp-nhp-$nm $L/nhp-$nm --vram 30 --cpu 8 --ram 40 $(aft $AFTER) -- $PY $PS/pp_eval.py --data lb_navhard --frames $F plans --models "$@" --tag p2op)
  e=$(sub fwp-nhe-$nm $L/nhe-$nm --vram 1 --cpu 4 --ram 16 --env OPI_ROOT=op_lb $(aft $p) -- \
      $JEV experiments/op_openloop/lib/op_interp.py nav-export --data lb_navhard --adapters base --plans $(for m in "$@"; do echo -n "$F@cinque_PP$m "; done))
  for m in "$@"; do
    ids+=("$(sub fwp-nhh-$F-$m $L/nhh-$F-$m --vram 1 --cpu 10 --ram 24 $(aft $e) -- \
      $NAV2 experiments/op_guard/scripts/nav_harness.py --poses $DATA_DIR/runs/op_lb/lb_navhard/preds/$F-cinque_PP${m}__base.npz --out $D/navhard/$F/$m --procs 10)")
  done; local IFS=,; echo "${ids[*]}"; }
hug_extract() { FULL_TAGS=$1 FULL_OUT=$D/report/hugsim $HPY $PS/pp_hugsim_report.py extract full || die "hugsim extract"; }

# ---------------------------------------------------------------- 1. prep + seed-0 round 1 (with preflight)
status "prep"
j=$(sub fwp-prep $L/prep --vram 16 --cpu 24 --ram 60 -- $PY $S/fw_p2.py prep train g0b g1s); waitdirs $L/prep
status "seed 0 round 1 collection (preflight first)"
c1s0=$(collect 0 1 P2-F-s0); waitdirs $L/col-xr1-s0-{0,1,2}
$PY $S/fw_p2report.py check || die "round-1 check"

# ---------------------------------------------------------------- 2. seed 0: rest of the DAgger chain, control, gate reads
status "seed 0: DAgger rounds 2-3, control, engine eval of P2"
tx0=$(dagger_rest 0 "$c1s0")
tc0=$(train PC-s0 C 0 none ${STEPS[3]})
ev=$(evalj P2-F-s0)
waitdirs $L/t-PX-s0 $L/t-PC-s0
status "seed 0 gate reads"
for t in PX-s0 PC-s0; do o=$(onnx $t); for pr in exam spec; do hug $pr $t $H12 12 "$o" >/dev/null; done; evalj $t >/dev/null; done
AFTER=""; p=$(nt_plans warp s0 PX-s0 PC-s0); AFTER=$p; nt_score warp s0 PX-s0 PC-s0 >/dev/null
AFTER=""; nh gimm s0 PX-s0 PC-s0 >/dev/null
waitdirs $L/nts-s0 $L/nhh-gimm-PX-s0 $L/nhh-gimm-PC-s0 $L/ev-P2-F-s0 $L/ev-PX-s0 $L/ev-PC-s0 \
  $(for t in PX-s0 PC-s0; do for pr in exam spec; do echo $L/h12-$pr-$t; done; done)
hug_extract P2-F-s0,P2-F-s1,PX-s0,PC-s0
$PY $S/fw_p2report.py gate; rc=$?
if [[ $rc == 3 ]]; then status "STOPPED at the gate (report/gate.md)"; date > "$D/STOPPED"; date > "$D/DONE"; exit 0; fi
[[ $rc == 0 ]] || die "gate script rc $rc"

# ---------------------------------------------------------------- 3. seed 1 and every readout
status "seed 1 chain + full readouts"
c1s1=$(collect 1 1 P2-F-s1)
tx1=$(dagger_rest 1 "$c1s1")
tc1=$(train PC-s1 C 1 none ${STEPS[3]})
for t in PX-s0 PC-s0; do for pr in exam spec; do hug $pr $t $H64 64 >/dev/null; done; done
for t in PX-s1 PC-s1; do
  dep=$([[ $t == PX-s1 ]] && echo "$tx1" || echo "$tc1")
  o=$(onnx $t "$dep"); for pr in exam spec; do hug $pr $t $H64 64 "$o" >/dev/null; done; evalj $t "$dep" >/dev/null
done
evalj P2-F-s1 >/dev/null
AFTER="$tx1,$tc1"
p=$(nt_plans warp s1 PX-s1 PC-s1); AFTER=$p; nt_score warp s1 PX-s1 PC-s1 >/dev/null
AFTER="$tx1,$tc1"
p=$(nt_plans gimm all P2-F-s0 P2-F-s1 PX-s0 PX-s1 PC-s0 PC-s1); AFTER=$p; nt_score gimm all P2-F-s0 P2-F-s1 PX-s0 PX-s1 PC-s0 PC-s1 >/dev/null
AFTER="$tx1,$tc1"; nh gimm s1 PX-s1 PC-s1 >/dev/null
AFTER="$tx1,$tc1"; nh warp all PX-s0 PX-s1 PC-s0 PC-s1 >/dev/null
waitdirs $L/nts-s1 $L/nts-all $(for m in PX-s1 PC-s1; do echo $L/nhh-gimm-$m; done) $(for m in PX-s0 PX-s1 PC-s0 PC-s1; do echo $L/nhh-warp-$m; done) \
  $(for t in PX-s0 PC-s0 PX-s1 PC-s1; do for pr in exam spec; do echo $L/h64-$pr-$t; done; done) $L/ev-PX-s1 $L/ev-PC-s1 $L/ev-P2-F-s1
hug_extract P2-F-s0,P2-F-s1,PX-s0,PX-s1,PC-s0,PC-s1
$PY $S/fw_p2report.py report || die "report"
status "done"
date > "$D/DONE"
