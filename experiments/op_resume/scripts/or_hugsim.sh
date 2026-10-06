#!/usr/bin/env bash
# One GPU-pool job of op_resume on HUGSIM: a resident Cinque server (shipped) and zs_run.py with interface preset `spec` on the
# scenarios of $SCEN, tag $TAG, agent opts $OPTS ('{}' = spec; '{"resume": {}}' = spec + the shared resume rule,
# jevdrive/openpilot/resume.py). Same server / zs_run calls as experiments/op_guard/scripts/guard_hugsim.sh plus --opts.
# Resumable (zs_run skips finished scenarios); fails unless every scenario has a non-crash row.
# Env: DATA_DIR GPU OUT SCEN TAG [OPTS='{}'] [WORKERS=5]. Submitted by or_submit.py.
set -euo pipefail
: "${DATA_DIR:?}" "${OUT:?}" "${SCEN:?}" "${TAG:?}"
cd "$(dirname "$0")/../../.."
source scripts/bench_lane.sh
bench_hugsim cinque spec "$TAG" "$SCEN" "${WORKERS:-5}" "" "${OPTS:-\{\}}"
