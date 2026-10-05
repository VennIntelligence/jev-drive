#!/usr/bin/env bash
# rc-*-pre chain (plans/2026-10-05-route-ft-prereg.md, 2026-10-06 section) on a leased card (lane rft-pre). STATUS / DONE / ERROR in $R/chain/<phase>/.
#   pre_chain.sh pilot <gpu> <cpus>     rc-bear-pre 400 steps (tag pilot-bear-pre), open-loop readouts with O, rc-bear-s0, rc-ctl-s0
#   pre_chain.sh full <gpu> <cpus>      rc-bear-pre and rc-ctl-pre (4000 steps) in parallel on the card, readouts, serving ONNX + adapter,
#                                       real-checkpoint equivalence for rc-bear-pre
set -uo pipefail
cd "$(dirname "$0")/../../.."
R=$DATA_DIR/runs/op_route_ft
PY=$DATA_DIR/envs/op-train/bin/python
phase=$1; gpu=$2; cpus=$3
D=$R/chain/pre-$phase; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
say() { echo "$(date '+%F %T') pre-$phase: $*" | tee -a "$D/log.txt" > /dev/null; echo "$(date '+%F %T') pre-$phase: $*" > "$D/STATUS"; }
die() { say "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
run() { CUDA_VISIBLE_DEVICES=$gpu taskset -c "$cpus" "$PY" experiments/op_route_ft/scripts/rft.py "$@"; }
case $phase in
  pilot)
    say "pilot rc-bear-pre 400 steps"; run train --arm rc-bear-pre --steps 400 --tag pilot-bear-pre --fresh >> "$D/log.txt" 2>&1 || die "pilot train"
    say "readouts"; run evalol --models O rc-bear-s0 rc-ctl-s0 pilot-bear-pre --carla ol >> "$D/log.txt" 2>&1 || die "evalol" ;;
  full)
    say "train rc-bear-pre + rc-ctl-pre"
    run train --arm rc-bear-pre --fresh > "$D/train-bear.log" 2>&1 & p1=$!
    run train --arm rc-ctl-pre --fresh > "$D/train-ctl.log" 2>&1 & p2=$!
    wait $p1 || die "train rc-bear-pre"; wait $p2 || die "train rc-ctl-pre"
    say "readouts"; run evalol --models rc-bear-pre-s0 rc-ctl-pre-s0 --carla ol >> "$D/log.txt" 2>&1 || die "evalol"
    say "onnx"
    $PY experiments/op_route_ft/scripts/route_onnx.py build --ckpt $R/runs/rc-bear-pre-s0/ckpt-final.pt --adapter $R/runs/rc-bear-pre-s0/adapter.npz \
      --out $R/onnx/rc-bear-pre-s0.onnx >> "$D/log.txt" 2>&1 || die "onnx bear"
    $PY experiments/op_route_ft/scripts/route_onnx.py build --ckpt $R/runs/rc-ctl-pre-s0/ckpt-final.pt --adapter none \
      --out $R/onnx/rc-ctl-pre-s0.onnx >> "$D/log.txt" 2>&1 || die "onnx ctl"
    say "equivalence rc-bear-pre"
    CUDA_VISIBLE_DEVICES=$gpu $PY experiments/op_route_ft/scripts/route_onnx.py ref --ckpt $R/runs/rc-bear-pre-s0/ckpt-final.pt \
      --adapter $R/runs/rc-bear-pre-s0/adapter.npz --out $R/onnx/rc-bear-pre-s0.ref.npz >> "$D/log.txt" 2>&1 || die "ref"
    CUDA_VISIBLE_DEVICES=$gpu $DATA_DIR/envs/openpilot/bin/python experiments/op_route_ft/scripts/route_onnx.py check --onnx $R/onnx/rc-bear-pre-s0.onnx \
      --ref $R/onnx/rc-bear-pre-s0.ref.npz >> "$D/log.txt" 2>&1 || die "check" ;;
  *) die "unknown phase $phase" ;;
esac
say "done"; date '+%F %T' > "$D/DONE"
