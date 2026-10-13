#!/usr/bin/env bash
# Tokyo: rf_pai_follow.py extract inside alpasim-base:0.89.0 (CPU only, no GPU, niced, few cores).
#   rf_pai_follow.sh <out pkl> LABEL=<run root> [LABEL=<run root> ...]    env: WORKERS (4), CPUS (4), LIMIT (0)
set -euo pipefail
OUT=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$(dirname "$OUT")"
exec nice -n 15 docker run --rm --cpus "${CPUS:-4}" --entrypoint bash \
  -v /data/datasets/nurec:/mnt/nre-data:ro -v /data/third_party/alpasim/src:/repo/src -v /data/third_party/alpasim/plugins:/repo/plugins \
  -v /data/runs:/data/runs:ro -v "$(dirname "$OUT")":/out -v "$HERE/rf_pai_follow.py":/tmp/rf_pai_follow.py:ro alpasim-base:0.89.0 \
  -c "umask 0000 && cd /repo && uv run --no-sync python /tmp/rf_pai_follow.py extract --workers ${WORKERS:-4} --limit ${LIMIT:-0} --out /out/$(basename "$OUT") --runs $*"
