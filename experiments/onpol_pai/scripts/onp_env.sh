#!/usr/bin/env bash
# Run a python script of experiments/onpol_pai in one of the two environments its stages need on the Tokyo box (CPU only, no network):
#   onp_env.sh [--drv] scripts/onp_rows.py <args>
#   default  image alpasim-base: AlpaSim's own venv (.asl reader, scene loader, polars) with the PAI driver's modules from $JEV
#   --drv    the driver image the rollouts were served from, with the same three lib files mounted over it as pai_run.sh mounts
# /data is mounted read-only; only $ONP_OUT and the staging dir $ONP_STAGE (host /tmp) are writable. Both images run as root with
# umask 0000 (alpasim-base's venv is root-only; pai_run.sh does the same): outputs are root-owned and world-writable.
# Env: DATA_DIR (/data), ALPASIM_SRC, JEV (repo snapshot the rollouts were served from), ONP_DEV (this directory's mirror), ONP_OUT,
#   ONP_STAGE, BASE, IMG.
set -euo pipefail
D=${DATA_DIR:-/data}; S=${ALPASIM_SRC:-$D/third_party/alpasim}; J=${JEV:-$D/runs/onpol_pai/code/pass1}
DEV=${ONP_DEV:-$D/runs/onpol_pai/dev}; OUT=${ONP_OUT:-$D/runs/onpol_pai/rows}; STAGE=${ONP_STAGE:-/tmp/onp_stage}
mkdir -p "$OUT" "$STAGE"
common=(--rm --network none --user 0:0 --entrypoint bash -e HOME=/tmp -e MPLCONFIGDIR=/tmp -e DATA_DIR="$D" -e ONP_STAGE="$STAGE" -e ONP_OUT="$OUT"
        -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENCV_FOR_THREADS_NUM=1 -e PYTHONUNBUFFERED=1
        -v /etc/localtime:/etc/localtime:ro -v "$D:$D:ro" -v "$OUT:$OUT" -v "$STAGE:$STAGE" -v "$DEV:/onp:ro")
if [[ ${1:-} == --drv ]]; then
  shift; py=$1; shift; lib=$J/experiments/alpasim/lib; A=/app/jev-drive
  exec docker run "${common[@]}" -e PYTHONPATH="$A/experiments/alpasim/lib:$A:/onp/lib" \
    -v "$lib/pai_core.py:$A/experiments/alpasim/lib/pai_core.py:ro" -v "$lib/pai_driver.py:$A/experiments/alpasim/lib/pai_driver.py:ro" \
    -v "$lib/serve_fix.py:$A/experiments/alpasim/lib/serve_fix.py:ro" -v "$J/jevdrive/openpilot/lead_long.py:$A/jevdrive/openpilot/lead_long.py:ro" \
    -e JEV_REPO="$A" -w /tmp "${IMG:-jev-alpasim:p2h10-f-s0-6f3d05d1}" -c 'umask 0000; exec python "$@"' py "/onp/$py" "$@"
fi
BASE=${BASE:-alpasim-base:$(sed -n 's/^version = "\(.*\)"/\1/p' "$S/pyproject.toml" | head -1)}
py=$1; shift
exec docker run "${common[@]}" -e ALPASIM_SRC="$S" -e JEV_REPO=/jev -e PYTHONPATH="$S/src/utils:/jev" -v "$J:/jev:ro" -w /repo "$BASE" \
  -c 'umask 0000; exec /repo/.venv/bin/python "$@"' py "/onp/$py" "$@"
