#!/usr/bin/env bash
# Lane DIAG1 (plans/2026-10-09-diag1-prereg.md): GPU stages (navtest extraction of shipped / adapted / switch variants; the SH30 switch variants
# on WOD val through the decision 155 harness) and the navtest swap scoring. Resumable: finished stages are skipped.
# Run: scripts/tmux_run.sh diag1 experiments/op_parity/scripts/diag1_chain.sh [stage ...]   (stages: nav wod score; default nav wod)
# State: $DATA_DIR/runs/op_parity/diag1/{STATUS, DONE, ERROR, log.txt, pool/}.
set -uo pipefail
cd "$(dirname "$0")/../../.."
D=$DATA_DIR/runs/op_parity/diag1; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
OP=$DATA_DIR/envs/openpilot/bin/python
J=$DATA_DIR/envs/jevdrive/bin/python
V=.venv/bin/python
CL="$J -m jevdrive.cl"
S=experiments/op_parity/scripts
L=$D/pool
ONNX=$DATA_DIR/runs/op_parity/hugsim/onnx
BIAS=$DATA_DIR/runs/op_parity/wod/diag
STAGES=${*:-nav wod}
status() { echo "$(date '+%F %T') diag1: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && return; rm -f "$ld/ERROR"
        $CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@" >/dev/null || die "submit $n"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 30; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; return 0; }
WV="zero biasmean biasresid posecv"
for st in $STAGES; do case $st in
nav)  status "navtest extraction"
      sub diag1-nav $L/nav --vram 24 --cpu 8 --ram 48 -- $PY $S/diag1.py nav-extract ;;
wod)  status "SH30 switch variants on WOD val"
      for s in 0 1; do t=SH30-F-s$s
        [[ -f $BIAS/bias-${t}_posecv.npz ]] || $PY $S/pp_wod_diag.py bias --arms $t --vars $WV || die "bias $t"
        sub diag1-wod-$t $L/wod-$t --vram 8 --cpu 14 --ram 40 -- $OP scripts/wod_zeroshot_openpilot.py --set rater extra --workers 12 \
            --onnx $ONNX/pp-$t.onnx --tag $(for v in $WV; do echo dx-${t}_$v; done) --bias $(for v in $WV; do echo $BIAS/bias-${t}_$v.npz; done)
      done ;;
score) status "navtest swap scoring"
      $V $S/diag1.py nav-swap || die "nav-swap"
      $V -m jevdrive.bench score-poses --poses $D/swap_poses.npz --out $D/swap_score.csv --traffic non_reactive --wait || die "score-poses" ;;
esac; done
for st in $STAGES; do case $st in
nav) waitdirs $L/nav ;;
wod) waitdirs $L/wod-SH30-F-s0 $L/wod-SH30-F-s1 ;;
esac; done
status "done: $STAGES"; date > "$D/DONE"
