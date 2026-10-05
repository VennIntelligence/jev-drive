#!/usr/bin/env bash
# rc-*-near staged closed-loop read of the 400-step pilots (user rule 2026-10-05: small set first, stop on a clear negative). Lease rft-near held.
#   near_pilot_cl.sh <gpu>   ONNX + adapter of pilot-bear-fix / pilot-bear-near, then near_lane.py routes=small (8 routes, 9 turns, desire off, curv)
# Candidates pilot-bear-fix / pilot-bear-near must be in experiments/op_guard/candidates.json. STATUS / DONE / ERROR in $R/chain/near-pilotcl/.
set -uo pipefail
cd "$(dirname "$0")/../../.."
R=$DATA_DIR/runs/op_route_ft; PY=$DATA_DIR/envs/op-train/bin/python; S=experiments/op_route_ft/scripts; gpu=$1
D=$R/chain/near-pilotcl; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
say() { echo "$(date '+%F %T') near-pilotcl: $*" | tee -a "$D/log.txt" > /dev/null; echo "$(date '+%F %T') near-pilotcl: $*" > "$D/STATUS"; }
die() { say "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
for t in pilot-bear-fix pilot-bear-near; do
  say "onnx $t"
  $PY $S/route_onnx.py build --ckpt $R/runs/$t/ckpt-final.pt --adapter $R/runs/$t/adapter.npz --out $R/onnx/$t.onnx >> "$D/log.txt" 2>&1 || die "onnx $t"
done
say "B2D small set"
.venv/bin/python -m jevdrive.cl run $S/near_lane.py --lane rft-near --root $R/near/cl-pilot --arg arms=pilot-bear-fix,pilot-bear-near \
  --arg routes=small --arg exec=curv >> "$D/log.txt" 2>&1 || die "lane"
say "done"; date '+%F %T' > "$D/DONE"
