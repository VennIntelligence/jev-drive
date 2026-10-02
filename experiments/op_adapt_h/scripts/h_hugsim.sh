#!/usr/bin/env bash
# Readout (d) of op-adapt H: an adapted checkpoint in HUGSIM closed loop on decision 96's 10 spin scenarios, PR #57 controller,
# no de-rotation rule (plans/2026-10-04-op-adapt-H-prereg.md). The checkpoint becomes a serving ONNX (op_adapt_l's op_l_onnx.py,
# no adapter), one resident Cinque server serves it, zs_run drives the scenarios; then h_hugsim_report.py counts spins.
# Usage (box, tmux):  GPU=<card> [WORKERS=3] experiments/op_adapt_h/scripts/h_hugsim.sh <run tag | O>
# Files: $DATA_DIR/runs/op_adapt_H/hugsim/<tag>/{STATUS,DONE,ERROR,results.csv,servers/}. The server is killed above 40 GB RSS.
set -uo pipefail
: "${DATA_DIR:?}" "${GPU:?}"
cd "$(dirname "$0")/../../.."
TAG=$1
H=$DATA_DIR/runs/op_adapt_H
OUT=$H/hugsim/$TAG
W=${WORKERS:-3}
HPY=$DATA_DIR/envs/hugsim/bin/python
mkdir -p "$OUT/servers" "$H/onnx"
rm -f "$OUT/servers/cinque.ready" "$OUT/DONE" "$OUT/ERROR"
fail() { echo "$(date +%T) $*" | tee "$OUT/ERROR"; exit 1; }
st() { echo "$(date +%T) $*" | tee "$OUT/STATUS"; }
ONNX=()
if [[ $TAG != O ]]; then
  [[ -f $H/onnx/$TAG.onnx ]] || CUDA_VISIBLE_DEVICES=$GPU "$DATA_DIR/envs/op-train/bin/python" experiments/op_adapt_l/scripts/op_l_onnx.py build \
      --ckpt "$H/runs/$TAG/ckpt-final.pt" --out "$H/onnx/$TAG.onnx" --no-adapter || fail "onnx build"
  if [[ ! -f $OUT/onnx_check.txt ]]; then                 # served ONNX vs the training port on op_adapt's stored WOD streams
    CUDA_VISIBLE_DEVICES=$GPU "$DATA_DIR/envs/op-train/bin/python" experiments/op_adapt_l/scripts/op_l_onnx.py ref \
        --ckpt "$H/runs/$TAG/ckpt-final.pt" --out "$OUT/onnx_ref.npz" || fail "onnx ref"
    CUDA_VISIBLE_DEVICES=$GPU "$DATA_DIR/envs/openpilot/bin/python" experiments/op_adapt_l/scripts/op_l_onnx.py check \
        --onnx "$H/onnx/$TAG.onnx" --ref "$OUT/onnx_ref.npz" > "$OUT/onnx_check.tmp" 2>&1 || fail "onnx check"
    grep "^stream" "$OUT/onnx_check.tmp" | awk '{ if ($16 + 0 > 0.5) bad = 1 } END { exit bad }' || fail "onnx differs from the port (plan xy max > 0.5 m)"
    mv "$OUT/onnx_check.tmp" "$OUT/onnx_check.txt"
  fi
  ONNX=(--onnx "$H/onnx/$TAG.onnx")
fi
CUDA_VISIBLE_DEVICES=$GPU setsid "$DATA_DIR/envs/openpilot/bin/python" -u experiments/hugsim/archive/hugsim_zs_server.py cinque "${ONNX[@]}" \
    --socket "$OUT/servers/cinque.sock" --ready-file "$OUT/servers/cinque.ready" > "$OUT/servers/cinque.log" 2>&1 &
srv=$!
trap 'kill -- -$srv 2>/dev/null' EXIT
( while kill -0 $srv 2>/dev/null; do                      # RSS watchdog (openpilot processes leaked before df2a38d)
    r=$(ps -o rss= -p $srv | tr -d ' '); [[ -n $r && $r -gt 41943040 ]] && { echo "$(date +%T) server RSS ${r} kB > 40 GB, killed" >> "$OUT/servers/watchdog.log"; kill -- -$srv; }
    sleep 30; done ) &
until [[ -f $OUT/servers/cinque.ready ]]; do sleep 5; kill -0 $srv 2>/dev/null || fail "server died"; done
st "server ready (pid $srv)"
$HPY experiments/hugsim/archive/zs_run.py setup-trees official fixed || fail "setup-trees"
st "10 spin scenarios"
$HPY experiments/hugsim/archive/zs_run.py run --out "$OUT" --agent cinque --controller fixed --gpu "$GPU" --workers "$W" \
    --scenarios experiments/hugsim/scripts/derot_spin10.txt --socket "$OUT/servers/cinque.sock" --opts '{}' --tag "cinque-fixed-H$TAG" \
    || fail "zs_run"
$HPY experiments/op_adapt_h/scripts/h_hugsim_report.py "$OUT" "cinque-fixed-H$TAG" || fail "report"
st "done"
touch "$OUT/DONE"
