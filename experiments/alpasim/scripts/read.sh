#!/usr/bin/env bash
# AlpaSim reads through queue_read.py (persistent stacks pulling scenes from a shared queue). Box, repo root:
#   bash experiments/alpasim/scripts/read.sh submit <queue dir> --name <job name> [--slots N] [--order read|scene] [--pool '<cl fanout flags>'] \
#        --read <out dir>=<scene list>=sh30=SH30_TAG:<tag>[,K:V] [--read ...] <wizard overrides>
#     -> prints the collect job id; each <out dir> then has run.sh's layout (aggregate/results-summary.json, rollouts/, driver-logs/s<k>/)
#   bash experiments/alpasim/scripts/read.sh slot <queue dir> <k>     (pool job body)
#   bash experiments/alpasim/scripts/read.sh collect <queue dir>      (pool job body, CPU)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); source "$here/env.sh"
exec "${ALPASIM_SRC:-$DATA_DIR/third_party/alpasim}/.venv/bin/python" "$here/queue_read.py" "$@"
