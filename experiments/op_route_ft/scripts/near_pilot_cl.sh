#!/usr/bin/env bash
# rc-*-near staged closed-loop read of the 400-step pilots (user rule 2026-10-05: small set first, stop on a clear negative).
#   near_pilot_cl.sh [GATE]   runs as a GPU-pool job: if GATE (an rc file) is given it must contain 0 (else skip, ERROR), then ONNX + adapter
#                             of pilot-bear-fix / pilot-bear-near, then near_lane.py --routes small --wait (8 routes, 9 turns, desire off, curv),
#                             whose units are pool jobs of their own. Submit it (the pool starts it once GATE exists):
#   .venv/bin/python -m jevdrive.cl submit --name near-pilotcl --vram 6 --when-exists /tmp/near_after.rc --log-dir $DATA_DIR/runs/op_route_ft/chain/near-pilotcl/pool \
#       -- bash experiments/op_route_ft/scripts/near_pilot_cl.sh /tmp/near_after.rc
# Candidates pilot-bear-fix / pilot-bear-near must be in experiments/op_guard/candidates.json. STATUS / DONE / ERROR in $R/chain/near-pilotcl/.
set -uo pipefail
cd "$(dirname "$0")/../../.."
R=$DATA_DIR/runs/op_route_ft; PY=$DATA_DIR/envs/op-train/bin/python; S=experiments/op_route_ft/scripts; gate=${1:-}
D=$R/chain/near-pilotcl; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
say() { echo "$(date '+%F %T') near-pilotcl: $*" | tee -a "$D/log.txt" > /dev/null; echo "$(date '+%F %T') near-pilotcl: $*" > "$D/STATUS"; }
die() { say "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
if [ -n "$gate" ]; then
  [ "$(cat "$gate" 2>/dev/null)" = 0 ] || { echo skip > /tmp/near_pcl.rc; die "skipped: $gate is '$(cat "$gate" 2>/dev/null)', not 0"; }
fi
for t in pilot-bear-fix pilot-bear-near; do
  say "onnx $t"
  $PY $S/route_onnx.py build --ckpt $R/runs/$t/ckpt-final.pt --adapter $R/runs/$t/adapter.npz --out $R/onnx/$t.onnx >> "$D/log.txt" 2>&1 || die "onnx $t"
done
say "B2D small set (pool units, logs in $R/nrp/<unit>/)"
.venv/bin/python $S/near_lane.py --root $R/nrp --arms pilot-bear-fix,pilot-bear-near --routes small --exec curv --wait >> "$D/log.txt" 2>&1 \
  || { echo 1 > /tmp/near_pcl.rc; die "B2D units (see $R/nrp/*/ERROR)"; }
echo 0 > /tmp/near_pcl.rc
say "done"; date '+%F %T' > "$D/DONE"
