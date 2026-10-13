#!/usr/bin/env bash
# Tokyo: hl_pack.py inside alpasim-base:0.89.0 (CPU only, niced, no GPU). Default arms / lists = batch sudden_turn_v1.
#   hl_pack.sh <pack dir under /data/runs/human_label> [extra hl_pack.py args]    env: CPUS (8), WORKERS (8)
set -euo pipefail
OUT=$(realpath -m "$1"); shift
HERE=$(cd "$(dirname "$0")" && pwd)
RN=/data/runs/route_night
mkdir -p "$OUT"
exec nice -n 15 docker run --rm --cpus "${CPUS:-8}" --entrypoint bash \
  -v /data/datasets/nurec:/mnt/nre-data:ro -v /data/third_party/alpasim/src:/repo/src -v /data/third_party/alpasim/plugins:/repo/plugins \
  -v /data/runs:/data/runs:ro -v "$OUT":"$OUT" -v "$HERE/hl_pack.py":/tmp/hl_pack.py:ro alpasim-base:0.89.0 \
  -c "umask 0000 && cd /repo && uv run --no-sync python /tmp/hl_pack.py --out $OUT --workers ${WORKERS:-8} \
      --arm rsg=$RN/r281/rsg --arm rb0=$RN/roll/rb0,$RN/roll/rb0_retry,$RN/r281/rb0 --arm rbsg2=$RN/r281/rbsg2 \
      --arm rsgpai=$RN/r281/rsgpai --arm ys0=/data/runs/onpol_long/pai281/ys0 \
      --lists $RN/train160.txt $RN/test281.txt --exclude $RN/exam147.txt $RN/exam_turn26.txt $*"
