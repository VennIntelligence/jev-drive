#!/usr/bin/env bash
# op_parity / wod-submit (plans/2026-10-08-wod-submit-rule.md): WLG on the 1 505 WOD-E2E test submission frames, then the seed combination, val check and statistics.
# Bias (CPU) -> serving through the GPU pool in three stages (1 target, 10, all) per seed -> predict (CPU). Rerunning resumes (served targets are skipped).
# State: $DATA_DIR/runs/op_parity/wod/submit/{STATUS, DONE, ERROR, chain.log, pool/}.
set -uo pipefail
cd "$(dirname "$0")/../../.."
D=$DATA_DIR/runs/op_parity/wod/submit; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/chain.log") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
OP=$DATA_DIR/envs/openpilot/bin/python
J=$DATA_DIR/envs/jevdrive/bin/python
CL="$J -m jevdrive.cl"
S=experiments/op_parity/scripts
ONNX=$DATA_DIR/runs/op_parity/hugsim/onnx
status() { echo "$(date '+%F %T') op_parity wod_submit: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
status "bias"
$PY $S/wod_submit.py bias || die "bias"
for lim in 1 10 0; do
  for t in WLG-full-s0 WLG-full-s1; do
    ld=$D/pool/$t-L$lim; rm -rf "$ld"; mkdir -p "$ld"
    status "serve $t limit $lim"
    $CL submit --owner op_parity --name wods-$t-L$lim --log-dir "$ld" --vram 8 --cpu 14 --ram 40 -- $OP scripts/wod_zeroshot_openpilot.py --set test \
        --workers 12 --limit $lim --onnx $ONNX/pp-$t.onnx --tag $t --bias $D/bias-test-$t.npz >/dev/null || die "submit $t L$lim"
    until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 20; done
    [[ -f $ld/ERROR ]] && die "serve $t L$lim: $(head -c 300 $ld/ERROR)"
  done
  [[ $lim == 1 ]] && status "stage 1 served; first test plans in preds/op_cinque_WLG-full-s*"
done
status "predict"
$J $S/wod_submit.py predict || die "predict"
status "done"; touch "$D/DONE"
