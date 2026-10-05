#!/usr/bin/env bash
# One GPU-pool job of op_resume on HUGSIM: a resident Cinque server (shipped) and zs_run.py with interface preset `spec` on the
# scenarios of $SCEN, tag $TAG, agent opts $OPTS ('{}' = spec; '{"resume": {}}' = spec + the shared resume rule,
# jevdrive/openpilot/resume.py). Same server / zs_run calls as experiments/op_guard/scripts/guard_hugsim.sh plus --opts.
# Resumable (zs_run skips finished scenarios); fails unless every scenario has a non-crash row.
# Env: DATA_DIR GPU OUT SCEN TAG [OPTS='{}'] [WORKERS=5]. Submitted by or_submit.py.
set -uo pipefail
: "${DATA_DIR:?}" "${GPU:?}" "${OUT:?}" "${SCEN:?}" "${TAG:?}"
cd "$(dirname "$0")/../../.."
HPY=$DATA_DIR/envs/hugsim/bin/python
OPTS=${OPTS:-'{}'}
SRV=$OUT/servers/$TAG
mkdir -p "$SRV"
rm -f "$SRV/ready"
CUDA_VISIBLE_DEVICES=$GPU setsid "$DATA_DIR/envs/openpilot/bin/python" -u experiments/hugsim/archive/hugsim_zs_server.py cinque \
    --socket "$SRV/sock" --ready-file "$SRV/ready" >> "$SRV/server.log" 2>&1 &
srv=$!
trap 'kill -- -$srv 2>/dev/null' EXIT
until [[ -f $SRV/ready ]]; do sleep 5; kill -0 $srv 2>/dev/null || { echo "server died (see $SRV/server.log)"; exit 1; }; done
echo "$(date +%T) server ready (pid $srv)"
$HPY experiments/hugsim/archive/zs_run.py setup-trees official fixed opctrl || exit 1
$HPY experiments/hugsim/archive/zs_run.py run --out "$OUT" --agent cinque --preset spec --gpu "$GPU" --workers "${WORKERS:-5}" \
    --scenarios "$SCEN" --socket "$SRV/sock" --tag "$TAG" --opts "$OPTS" || exit 1
$HPY - "$OUT/results.csv" "$SCEN" "$TAG" <<'PY'
import csv, sys
from pathlib import Path
want = {Path(s).stem for s in open(sys.argv[2]).read().split()}
got = {r["scenario"] for r in csv.DictReader(open(sys.argv[1])) if r["tag"] == sys.argv[3] and r["end"] != "crash"}
miss = sorted(want - got)
print("missing:", miss) if miss else print("all %d scenarios done" % len(want))
sys.exit(1 if miss else 0)
PY
