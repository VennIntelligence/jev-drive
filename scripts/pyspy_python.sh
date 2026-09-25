#!/usr/bin/env bash
# Route-process interpreter for `b2d_run.py --python`: runs b2d_route.py under py-spy, which as the parent may trace
# its child under ptrace_scope=1. Collapsed stacks (with thread ids) land next to the attempt, in <--out>/pyspy.txt.
# Profiling only: sampling costs the route process some CPU (docs/bench2drive-cost.md, 2026-09-25).
out=. prev=
for a in "$@"; do [[ $prev == --out ]] && out=$a; prev=$a; done
mkdir -p "$out"
exec "$DATA_DIR/envs/carla/bin/py-spy" record --nonblocking --threads -r "${PYSPY_RATE:-50}" -f raw \
    -o "$out/pyspy.txt" ${PYSPY_ARGS:-} -- "$DATA_DIR/envs/carla/bin/python" "$@"
