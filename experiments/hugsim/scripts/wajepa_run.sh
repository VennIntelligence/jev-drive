#!/usr/bin/env bash
# One GPU-pool job: WA-JEPA (shipped client, experiments/hugsim/scripts/wajepa_e2e.sh) on a scenario list under the PR #57 controller
# (tree `fixed`), tag `wajepa`. Resumable: finished scenarios in <out>/results.csv are skipped.
#   python -m jevdrive.cl submit --name wajepa-s1 --vram 55 --cpu 12 --log-dir $DATA_DIR/runs/hugsim-wajepa/pool/s1 -- \
#       bash experiments/hugsim/scripts/wajepa_run.sh <list.txt> [workers] [out_dir]
# TREE=fixedc: tree fixed + patches/hugsim/optional/command-index-clamp.patch (only for a scenario whose route-end step crashes the simulator).
# Lists: experiments/hugsim/scripts/wajepa_smoke.txt (1), derot_spin10.txt (10 PR #57 Cinque spinners), derot_all64.txt (the exam's 64).
set -uo pipefail
: "${DATA_DIR:?}" "${CL_GPU:?run inside a pool job}"
cd "$(dirname "$0")/../../.."
OUT=${3:-$DATA_DIR/runs/hugsim-wajepa}
HPY=$DATA_DIR/envs/hugsim/bin/python
mkdir -p "$OUT"
$HPY experiments/hugsim/archive/zs_run.py setup-trees fixed ${TREE:+$TREE} || exit 1
exec $HPY experiments/hugsim/archive/zs_run.py run --preset exam --out "$OUT" --agent wajepa --controller ${TREE:-fixed} --gpu "$CL_GPU" \
    --workers "${2:-4}" --scenarios "$1" --tag wajepa --timeout 5400 --retries 0
