# sh30 with the serve-side static hold (experiments/alpasim/lib/sh30_hold.py wrapping sh30_driver.py; JEV_HOLD from the environment).
# Sourced by run.sh ($out $SRC $here $DATA_DIR are set); the same environment as run.sh's sh30 case, only the entry file differs.
cwd=$here/..
cmd="exec env ALPASIM_SRC=$SRC TMPDIR=$out/driver-tmp TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$DATA_DIR/cache/torchinductor} ALPASIM_DRIVER_LOG_DIR=$out/driver-logs ALPASIM_DRIVER_GRPC_WORKERS=${ALPASIM_DRIVER_GRPC_WORKERS:-8} \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 SH30_TAG=${SH30_TAG:-SH30-F-s0} SH30_COLD=${SH30_COLD:-backwarp} \
SH30_DUMP=${SH30_DUMP:-0} SH30_LHT=${SH30_LHT:-0} $DATA_DIR/envs/op-train/bin/python $here/../lib/sh30_hold.py"
