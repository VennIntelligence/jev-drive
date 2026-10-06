# Source from legacy HUGSIM orchestrators after changing to the repo root. This is an input/output adapter, not a runner.
# bench_hugsim MODEL PRESET TAG SCENARIOS WORKERS [CONTROLLER] [OPTS]
B=("$PWD/.venv/bin/python" -m jevdrive.bench)
bench_hugsim() {
    local model=$1 preset=$2 tag=$3 scenarios=$4 workers=$5 controller=${6:-} opts=${7:-\{\}}
    local cfg repeat
    cfg=$("${B[0]}" - <<'PY'
import json, os
from jevdrive.bench.hugsim import CONTROLLER_ENV
print(json.dumps({k: json.loads(os.environ[k]) for k in CONTROLLER_ENV if os.environ.get(k)}))
PY
    ) || return
    repeat=${BENCH_REPEAT:-}
    if [[ -z $repeat ]]; then
    repeat=$("${B[0]}" - "$OUT" "$tag" <<'PY'
import hashlib, sys
from pathlib import Path
print(hashlib.sha256((str(Path(sys.argv[1]).resolve()) + '/' + sys.argv[2]).encode()).hexdigest()[:12])
PY
    ) || return
    fi
    local mode=()
    [[ -n ${CL_POOL_JOB:-} ]] && mode=(--in-pool)
    [[ -n ${BENCH_WAIT_TIMEOUT_S:-} ]] && mode+=(--wait-timeout-s "$BENCH_WAIT_TIMEOUT_S")
    [[ -n ${TIMEOUT:-} ]] && mode+=(--timeout-s "$TIMEOUT")
    [[ -n ${RETRIES:-} ]] && mode+=(--retries "$RETRIES")
    "${B[@]}" run --model "$model" --bench hugsim --preset "$preset" --controller "$controller" --opts "$opts" \
        --controller-env "$cfg" --onnx "${ONNX:-}" --repeat "$repeat" --scenarios "$scenarios" --workers "$workers" --wait \
        --publish-out "$OUT" --publish-tag "$tag" "${mode[@]}"
}
