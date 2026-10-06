#!/usr/bin/env bash
# CPU scoring chain (tmux jev, scripts/tmux_run.sh): wait for the plans job, then v2 EPDMS of every model and of the shipped ONNX run.
#   experiments/op_parity/scripts/pp_score_chain.sh <plans pool log dir> <models ...>
set -uo pipefail
wait_dir=$1; shift
until [[ -e $wait_dir/DONE || -e $wait_dir/ERROR ]]; do sleep 15; done
[[ -e $wait_dir/ERROR ]] && { echo "plans job failed: $wait_dir/ERROR"; exit 1; }
cd "$(dirname "$0")/../../.."
MODELS=()
for m in "$@"; do MODELS+=("${m%%:*}@${FRAMES:-gimm}${m#${m%%:*}}"); done
exec .venv/bin/python -m jevdrive.bench run --model "${MODELS[@]}" cinque --bench navtest --wait
