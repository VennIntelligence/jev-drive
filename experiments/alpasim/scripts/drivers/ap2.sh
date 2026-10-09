# AP2 driver (experiments/alpasim/lib/ap2_driver.py, envs/op-train): openpilot Cinque + ego adapter trained for AlpaSim's inputs.
# Sourced by run.sh ($out $SRC $here $DATA_DIR are set); AP2_TAG / AP2_COLD / SH30_DUMP / SH30_LHT pass through.
cwd=$here/..
cmd="exec env ALPASIM_SRC=$SRC TMPDIR=$out/driver-tmp TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$DATA_DIR/cache/torchinductor} ALPASIM_DRIVER_LOG_DIR=$out/driver-logs ALPASIM_DRIVER_GRPC_WORKERS=8 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 AP2_TAG=${AP2_TAG:-AP2-AB-s0} AP2_COLD=${AP2_COLD:-} \
SH30_DUMP=${SH30_DUMP:-0} SH30_LHT=${SH30_LHT:-0} $DATA_DIR/envs/op-train/bin/python $here/../lib/ap2_driver.py"
