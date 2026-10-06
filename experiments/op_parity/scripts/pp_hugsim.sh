#!/usr/bin/env bash
# op_parity arms in the HUGSIM harness of experiments/hugsim/results/wajepa_ref.md: Cinque under the PR #57 controller (tree `fixed`),
# preset exam, opts as the 2026-09-25 `cinque-fixed` run ({} + the scene's traffic convention), plus opt `parity` (lib/parity_hugsim.py).
# Standard arms use bench stages; leased callers reuse their job. Equivalence keeps its diagnostic servers here.
#   equiv                 equivalence tests 1-3 (results/hugsim_harness.md) -> $R/equiv/*.json
#   arm TAG LIST [W]      TAG = P0 (shipped weights + an untrained P3 adapter: bias exactly 0, full input path) | P2-init | P3-init |
#                         a pp_train run tag (P1-s0, P2-s0, P3-s0); zs_run tag pp-<TAG>, resumable; results in $R/results.csv
#                         env PRESET=spec: interface preset `spec` instead (tree opctrl: openpilot's lateral path, action curvature ->
#                         clip_curvature -> lateralDelay 0.25 s; decisions 118 / 124), zs_run tag pp-spec-<TAG>
#                         PRESET=spec_plan: spec with the curvature of the model's own plan (modeld get_curvature_from_plan), tag pp-specplan-<TAG>
#                         PRESET=spec_plan_smooth (mean plan curvature over 0.5-1.5 s) / spec_plan_mpc (legacy lateral MPC, lib/op_lat_mpc.py): tags pp-specplansmooth- / pp-specplanmpc-
#                         env TIMEOUT (s per scenario, default 5400) and RETRIES (default 1)
#   python -m jevdrive.cl submit --name pp-hug-P2 --vram 40 --cpu 12 --log-dir $DATA_DIR/runs/op_parity/hugsim/pool/P2-s0 -- \
#       bash experiments/op_parity/scripts/pp_hugsim.sh arm P2-s0 experiments/hugsim/scripts/derot_spin10.txt 4
set -uo pipefail
: "${DATA_DIR:?}"
cd "$(dirname "$0")/../../.."
R=$DATA_DIR/runs/op_parity/hugsim
S=experiments/op_parity/scripts/pp_hugsim.py
TPY=$DATA_DIR/envs/op-train/bin/python
OPY=$DATA_DIR/envs/openpilot/bin/python
HPY=$DATA_DIR/envs/hugsim/bin/python
mkdir -p "$R/servers" "$R/onnx" "$R/equiv"
PIDS=()
trap 'for p in "${PIDS[@]}"; do kill -- -$p 2>/dev/null; done' EXIT
fail() { echo "$(date +%T) $*"; exit 1; }
onnx_of() { case $1 in P0|*-init) echo "$R/onnx/pp-shipped.onnx";; *) echo "$R/onnx/pp-$1.onnx";; esac; }
build() {  # tag -> onnx (skipped if present)
    local o; o=$(onnx_of "$1")
    [[ -f $o ]] || $TPY $S onnx --tag "$1" --out "$o" || fail "onnx $1"
}
start() {  # name ready-file cmd...
    local name=$1 ready=$2; shift 2
    rm -f "$ready"
    setsid "$@" > "$R/servers/$name.log" 2>&1 &
    local p=$!
    PIDS+=("$p")
    until [[ -f $ready ]]; do sleep 5; kill -0 "$p" 2>/dev/null || fail "$name died: $(tail -5 "$R/servers/$name.log")"; done
    echo "$(date +%T) $name ready"
}
bias_server() {  # name tag
    start "$1" "$R/servers/$1.ready" $TPY -u $S serve --tag "$2" --socket "$R/servers/$1.sock" --ready-file "$R/servers/$1.ready"
}
op_server() {  # name onnx
    start "$1" "$R/servers/$1.ready" $OPY -u experiments/hugsim/archive/hugsim_zs_server.py cinque --onnx "$2" \
        --socket "$R/servers/$1.sock" --ready-file "$R/servers/$1.ready"
}

case ${1:?equiv | arm} in
equiv)
    : "${CL_GPU:?run equivalence inside a pool job}"
    E=$R/equiv
    build P0
    [[ -f $E/synth/ckpt-final.pt ]] || $TPY $S synth --out "$E/synth" || fail synth
    [[ -f $E/synth.onnx ]] || $TPY $S onnx --tag "$E/synth/ckpt-final.pt" --out "$E/synth.onnx" || fail "synth onnx"
    for t in P0 P2-init P3-init; do $TPY $S ref --tag $t --out "$E/ref-$t.npz" || fail "ref $t"; done
    $TPY $S ref --tag "$E/synth/ckpt-final.pt" --out "$E/ref-synth.npz" || fail "ref synth"
    for t in P2-init P3-init; do
        bias_server "eq-$t" $t
        $OPY $S check --onnx "$(onnx_of P0)" --ref "$E/ref-$t.npz" --socket "$R/servers/eq-$t.sock" --base cinque --json "$E/check-$t.json" || fail "check $t"
    done
    bias_server eq-P0 P0
    $OPY $S check --onnx "$(onnx_of P0)" --ref "$E/ref-P0.npz" --socket "$R/servers/eq-P0.sock" --json "$E/check-P0.json" || fail "check P0"
    bias_server eq-synth "$E/synth/ckpt-final.pt"
    for be in trt cuda-iob; do
        $OPY $S check --onnx "$E/synth.onnx" --ref "$E/ref-synth.npz" --socket "$R/servers/eq-synth.sock" --backend $be \
            --json "$E/check-synth-$be.json" || fail "check synth $be"
    done
    echo "$(date +%T) equiv done";;
arm)
    TAG=${2:?tag}; LIST=${3:?scenario list}; W=${4:-4}
    PRESET=${PRESET:-exam}
    PFX=pp-; [[ $PRESET != exam ]] && PFX=pp-${PRESET//_/}-
    mode=(); [[ -n ${CL_POOL_JOB:-} ]] && mode=(--in-pool)
    [[ -n ${TIMEOUT:-} ]] && mode+=(--timeout-s "$TIMEOUT")
    [[ -n ${RETRIES:-} ]] && mode+=(--retries "$RETRIES")
    "$PWD/.venv/bin/python" -m jevdrive.bench run --model "$TAG" --bench hugsim --preset "$PRESET" \
        --scenarios "$LIST" --workers "$W" --wait --publish-out "$R" --publish-tag "$PFX$TAG" "${mode[@]}" || fail "bench hugsim $TAG"
    echo "$(date +%T) arm $TAG done";;
*) fail "unknown mode $1";;
esac
