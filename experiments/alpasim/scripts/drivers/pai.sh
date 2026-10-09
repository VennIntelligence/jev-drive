# PAI-track driver (experiments/alpasim/lib/pai_driver.py: the SH30 family served on the PAI simulator's cameras, envs/op-train).
# Sourced by run.sh ($out $SRC $here $DATA_DIR are set). Env passthrough: SH30_TAG (checkpoint, default P2H10-F-s0), PAI_CAM / PAI_COLD /
# PAI_EVERY / PAI_DUMP / PAI_LHT (pai_driver.py), JEV_VCONT / JEV_LEAD (serve_fix.py, off by default), DRV_PY (a variant in lib/ run instead
# of pai_driver.py, e.g. col1_pai_driver.py).
cwd=$here/..
cmd="exec env ALPASIM_SRC=$SRC TMPDIR=$out/driver-tmp TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$DATA_DIR/cache/torchinductor} ALPASIM_DRIVER_LOG_DIR=$out/driver-logs ALPASIM_DRIVER_GRPC_WORKERS=${ALPASIM_DRIVER_GRPC_WORKERS:-8} \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 SH30_TAG=${SH30_TAG:-P2H10-F-s0} PAI_DUMP=${PAI_DUMP:-0} ${PAI_CAM:+PAI_CAM=$PAI_CAM} ${PAI_COLD:+PAI_COLD=$PAI_COLD} \
${PAI_EVERY:+PAI_EVERY=$PAI_EVERY} ${PAI_LHT:+PAI_LHT=$PAI_LHT} ${JEV_VCONT:+JEV_VCONT=$JEV_VCONT} ${JEV_LEAD:+JEV_LEAD=$JEV_LEAD} \
$DATA_DIR/envs/op-train/bin/python $here/../lib/${DRV_PY:-pai_driver.py}"
