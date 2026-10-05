#!/usr/bin/env bash
# HUGSIM job of the guard line `hugsim` (cllib.hugsim_unit, one GPU-pool job): one resident Cinque server (the candidate's
# serving ONNX if $ONNX is set, shipped otherwise) and zs_run.py with interface preset `spec` on the scenarios of $SCEN, tag $TAG.
# Resumable (zs_run skips finished scenarios). Env: DATA_DIR GPU OUT SCEN TAG [ONNX] [WORKERS=5].
# Same server / zs_run calls as experiments/leaderboard_audit/scripts/unified_hugsim_chain.sh (arm d118 = spec) and
# experiments/op_adapt_h/scripts/h_hugsim.sh (--onnx); the server is killed above 40 GB RSS.
set -uo pipefail
: "${DATA_DIR:?}" "${GPU:?}" "${OUT:?}" "${SCEN:?}" "${TAG:?}"
cd "$(dirname "$0")/../../.."
HPY=$DATA_DIR/envs/hugsim/bin/python
mkdir -p "$OUT/servers"
rm -f "$OUT/servers/cinque.ready"
CUDA_VISIBLE_DEVICES=$GPU setsid "$DATA_DIR/envs/openpilot/bin/python" -u experiments/hugsim/archive/hugsim_zs_server.py cinque ${ONNX:+--onnx "$ONNX"} \
    --socket "$OUT/servers/cinque.sock" --ready-file "$OUT/servers/cinque.ready" >> "$OUT/servers/cinque.log" 2>&1 &
srv=$!
trap 'kill -- -$srv 2>/dev/null' EXIT
( while kill -0 $srv 2>/dev/null; do
    r=$(ps -o rss= -p $srv | tr -d ' '); [[ -n $r && $r -gt 41943040 ]] && { echo "$(date +%T) server RSS ${r} kB > 40 GB, killed" >> "$OUT/servers/watchdog.log"; kill -- -$srv; }
    sleep 30; done ) &
until [[ -f $OUT/servers/cinque.ready ]]; do sleep 5; kill -0 $srv 2>/dev/null || { echo "server died (see $OUT/servers/cinque.log)"; exit 1; }; done
echo "$(date +%T) server ready (pid $srv, onnx ${ONNX:-shipped})"
$HPY experiments/hugsim/archive/zs_run.py setup-trees official fixed opctrl || exit 1
$HPY experiments/hugsim/archive/zs_run.py run --out "$OUT" --agent cinque --preset spec --gpu "$GPU" --workers "${WORKERS:-5}" \
    --scenarios "$SCEN" --socket "$OUT/servers/cinque.sock" --tag "$TAG"
