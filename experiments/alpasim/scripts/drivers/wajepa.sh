# Sourced by run.sh: WA-JEPA driver (experiments/alpasim/lib/wajepa_driver.py) in envs/wajepa, cwd = the WA-JEPA checkout.
# Env passthrough: WAJ_COLD (repeat | cv), WAJ_DUMP (sessions whose fed frames are saved), WAJ_CFG / WAJ_CKPT.
WJ=$DATA_DIR/third_party/wajepa
cwd=$WJ
cmd="exec env ALPASIM_SRC=$SRC TMPDIR=$out/driver-tmp ALPASIM_DRIVER_LOG_DIR=$out/driver-logs ALPASIM_DRIVER_GRPC_WORKERS=8 \
PYTHONPATH=$WJ:$DATA_DIR/third_party/navsim OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
WAJ_COLD=${WAJ_COLD:-repeat} WAJ_DUMP=${WAJ_DUMP:-0} $DATA_DIR/envs/wajepa/bin/python $here/../lib/wajepa_driver.py"
