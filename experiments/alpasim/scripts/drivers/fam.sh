# Sourced by run.sh: the sh30 driver (as `run.sh <dir> sh30`) through experiments/alpasim_family/scripts/fam_driver.py, which applies FAM_ARM
# (a perturbed CAM_F0 calibration or image; fam_perturb.py). SH30_TAG / SH30_COMPILE / SH30_DUMP / FAM_ARM / FAM_CALIB pass through.
cwd=$here/..
cmd="exec env ALPASIM_SRC=$SRC TMPDIR=$out/driver-tmp TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$DATA_DIR/cache/torchinductor} ALPASIM_DRIVER_LOG_DIR=$out/driver-logs ALPASIM_DRIVER_GRPC_WORKERS=${ALPASIM_DRIVER_GRPC_WORKERS:-8} \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 SH30_TAG=${SH30_TAG:-YR10m10-F-s0} SH30_COLD=${SH30_COLD:-backwarp} SH30_COMPILE=${SH30_COMPILE:-1} \
SH30_DUMP=${SH30_DUMP:-0} SH30_LHT=0 FAM_ARM='${FAM_ARM:-}' FAM_CALIB=${FAM_CALIB:-} $DATA_DIR/envs/op-train/bin/python $here/../../alpasim_family/scripts/fam_driver.py"
