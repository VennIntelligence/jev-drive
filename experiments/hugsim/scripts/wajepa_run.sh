#!/usr/bin/env bash
# One GPU-pool job: WA-JEPA (shipped client, experiments/hugsim/scripts/wajepa_e2e.sh) on a scenario list under the PR #57 controller
# (tree `fixed`), tag `wajepa`. Resumable: finished scenarios in <out>/results.csv are skipped.
#   python -m jevdrive.cl submit --name wajepa-s1 --vram 55 --cpu 12 --log-dir $DATA_DIR/runs/hugsim-wajepa/pool/s1 -- \
#       bash experiments/hugsim/scripts/wajepa_run.sh <list.txt> [workers] [out_dir]
# TREE=fixedc: tree fixed + patches/hugsim/optional/command-index-clamp.patch (only for a scenario whose route-end step crashes the simulator).
# Lists: experiments/hugsim/scripts/wajepa_smoke.txt (1), derot_spin10.txt (10 PR #57 Cinque spinners), derot_all64.txt (the exam's 64).
set -euo pipefail
: "${DATA_DIR:?}"
cd "$(dirname "$0")/../../.."
OUT=${3:-$DATA_DIR/runs/hugsim-wajepa}
source scripts/bench_lane.sh
bench_hugsim WA-JEPA exam wajepa "$1" "${2:-4}" "${TREE:-fixed}"
