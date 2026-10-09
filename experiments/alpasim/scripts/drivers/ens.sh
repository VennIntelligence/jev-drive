# Adapter ensemble driver (experiments/alpasim/lib/ens_driver.py, envs/op-train): N checkpoints of one input standard on one shared encoder pass.
# Sourced by run.sh ($out $SRC $here $DATA_DIR are set); ENS_TAGS=tag1,tag2 (required), ENS_COLD / SH30_DUMP / SH30_LHT / SH30_MOTION pass through.
cwd=$here/..
cmd="exec env ALPASIM_SRC=$SRC TMPDIR=$out/driver-tmp ALPASIM_DRIVER_LOG_DIR=$out/driver-logs ALPASIM_DRIVER_GRPC_WORKERS=8 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ENS_TAGS=${ENS_TAGS:?ENS_TAGS=tag1,tag2} ENS_COLD=${ENS_COLD:-} \
SH30_DUMP=${SH30_DUMP:-0} SH30_LHT=${SH30_LHT:-0} $DATA_DIR/envs/op-train/bin/python $here/../lib/ens_driver.py"
