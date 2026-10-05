#!/usr/bin/env bash
# op_parity arms in the HUGSIM harness of experiments/hugsim/results/wajepa_ref.md: Cinque under the PR #57 controller (tree `fixed`),
# preset exam, opts as the 2026-09-25 `cinque-fixed` run ({} + the scene's traffic convention), plus opt `parity` (lib/parity_hugsim.py).
# One GPU-pool job per call (needs CL_GPU); every server is started here and killed on exit.
#   equiv                 equivalence tests 1-3 (results/hugsim_harness.md) -> $R/equiv/*.json
#   arm TAG LIST [W]      TAG = P0 (shipped weights + an untrained P3 adapter: bias exactly 0, full input path) | P2-init | P3-init |
#                         a pp_train run tag (P1-s0, P2-s0, P3-s0); zs_run tag pp-<TAG>, resumable; results in $R/results.csv
#   python -m jevdrive.cl submit --name pp-hug-P2 --vram 40 --cpu 12 --log-dir $DATA_DIR/runs/op_parity/hugsim/pool/P2-s0 -- \
#       bash experiments/op_parity/scripts/pp_hugsim.sh arm P2-s0 experiments/hugsim/scripts/derot_spin10.txt 4
set -uo pipefail
: "${DATA_DIR:?}" "${CL_GPU:?run inside a pool job}"
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
    SRV=$TAG; [[ $TAG == P0 ]] && SRV=P3-init
    build "$TAG"
    bias_server "b-$TAG" "$SRV"
    op_server "op-$TAG" "$(onnx_of "$TAG")"
    $HPY experiments/hugsim/archive/zs_run.py setup-trees fixed || fail setup-trees
    $HPY experiments/hugsim/archive/zs_run.py run --preset exam --out "$R" --agent cinque --controller fixed --gpu "$CL_GPU" \
        --workers "$W" --scenarios "$LIST" --socket "$R/servers/op-$TAG.sock" --tag "pp-$TAG" --timeout 5400 --retries 1 \
        --opts "{\"parity\": {\"socket\": \"$R/servers/b-$TAG.sock\"}}" || fail "zs_run $TAG"
    echo "$(date +%T) arm $TAG done";;
*) fail "unknown mode $1";;
esac
