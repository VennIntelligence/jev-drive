# Sourced by run.sh: the shipped SimScale DiffusionDrive sample driver, run natively in envs/alpasim-ltf (same pinned requirements as the
# LTF sample; torch 2.8 instead of 2.6 for sm_120). The ENV block of the sample's Dockerfile, write dirs moved off the system disk.
# Checkpoint: $DATA_DIR/models/alpasim_dd/diffusiondrive_sim_navhard.ckpt, staged by the sample's prepare_assets.sh (size + sha256 checked).
cwd=$SRC/e2e_challenge/sample_submission_simscale_navsim_diffusiondrive
cmd="exec env TMPDIR=$out/driver-tmp XDG_CACHE_HOME=$out/driver-tmp/cache TORCH_HOME=$out/driver-tmp/torch \
HF_HOME=$out/driver-tmp/hf HF_HUB_OFFLINE=1 MPLCONFIGDIR=$out/driver-tmp/mpl CUDA_CACHE_PATH=$out/driver-tmp/nv \
NUMBA_CACHE_DIR=$out/driver-tmp/numba TORCH_EXTENSIONS_DIR=$out/driver-tmp/torch_ext ALPASIM_DRIVER_LOG_DIR=$out/driver-logs \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1 TORCH_NUM_INTEROP_THREADS=1 \
ALPASIM_DRIVER_GRPC_WORKERS=4 DIFFUSIONDRIVE_CHECKPOINT_PATH=$DATA_DIR/models/alpasim_dd/diffusiondrive_sim_navhard.ckpt DIFFUSIONDRIVE_DEVICE=cuda \
DIFFUSIONDRIVE_MAX_BATCH_SIZE=${DD_MAX_BATCH_SIZE:-1} DIFFUSIONDRIVE_BATCH_WINDOW_MS=2 $DATA_DIR/envs/alpasim-ltf/bin/python -m navsim_diffusiondrive_challenge.driver"
