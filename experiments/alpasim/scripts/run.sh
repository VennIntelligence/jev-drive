#!/usr/bin/env bash
# One AlpaSim closed-loop run as plain processes (run_native.py) with a shipped driver. Submit it to the GPU pool:
#   python -m jevdrive.cl submit --name alpasim-dev --vram 30 --cpu 12 --log-dir <run dir>/pool -- \
#     bash experiments/alpasim/scripts/run.sh <run dir> <starter|ltf|sh30|drivers/*.sh name> [--tap] +e2e_challenge_nuplan=dev [overrides]
# sh30 = our driver (experiments/alpasim/lib/sh30_driver.py, envs/op-train); SH30_TAG / SH30_COLD / SH30_DUMP / SH30_LHT / SH30_SYNTH /
# ALPASIM_DRIVER_GRPC_WORKERS pass through.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); source "$here/env.sh"
out=$1 drv=$2; shift 2
SRC=${ALPASIM_SRC:-$DATA_DIR/third_party/alpasim}
mkdir -p "$out/driver-tmp"
case $drv in
  starter) cwd=$SRC; cmd="exec $SRC/.venv/bin/python e2e_challenge/starter_kit/driver.py" ;;
  ltf)     # the ENV block of the sample's Dockerfile, with write dirs moved off the system disk
    cwd=$SRC/e2e_challenge/sample_submission_simscale_navsim_transfuser
    cmd="exec env TMPDIR=$out/driver-tmp XDG_CACHE_HOME=$out/driver-tmp/cache TORCH_HOME=$out/driver-tmp/torch \
HF_HOME=$out/driver-tmp/hf HF_HUB_OFFLINE=1 MPLCONFIGDIR=$out/driver-tmp/mpl CUDA_CACHE_PATH=$out/driver-tmp/nv \
NUMBA_CACHE_DIR=$out/driver-tmp/numba TORCH_EXTENSIONS_DIR=$out/driver-tmp/torch_ext ALPASIM_DRIVER_LOG_DIR=$out/driver-logs \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1 TORCH_NUM_INTEROP_THREADS=1 \
ALPASIM_DRIVER_GRPC_WORKERS=4 LTF_CHECKPOINT_PATH=$DATA_DIR/models/alpasim_ltf/ltf_sim_navtest.ckpt LTF_DEVICE=cuda \
LTF_MAX_BATCH_SIZE=${LTF_MAX_BATCH_SIZE:-2} LTF_BATCH_WINDOW_MS=2 $DATA_DIR/envs/alpasim-ltf/bin/python -m navsim_transfuser_challenge.driver" ;;
  sh30)
    cwd=$here/..
    cmd="exec env ALPASIM_SRC=$SRC TMPDIR=$out/driver-tmp TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$DATA_DIR/cache/torchinductor} ALPASIM_DRIVER_LOG_DIR=$out/driver-logs ALPASIM_DRIVER_GRPC_WORKERS=${ALPASIM_DRIVER_GRPC_WORKERS:-8} \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 SH30_TAG=${SH30_TAG:-SH30-F-s0} SH30_COLD=${SH30_COLD:-backwarp} \
SH30_DUMP=${SH30_DUMP:-0} SH30_LHT=${SH30_LHT:-0} $DATA_DIR/envs/op-train/bin/python $here/../lib/sh30_driver.py" ;;
  *)       # any other driver: scripts/drivers/<name>.sh, sourced here, sets cwd and cmd (it sees $out $SRC $here $DATA_DIR)
    [[ -f $here/drivers/$drv.sh ]] || { echo "unknown driver $drv" >&2; exit 2; }
    source "$here/drivers/$drv.sh" ;;
esac
exec "$SRC/.venv/bin/python" "$here/run_native.py" --log-dir "$out" --driver "$cmd" --driver-cwd "$cwd" "$@"
