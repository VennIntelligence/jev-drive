# Sourced by run.sh: the shipped LTF sample exactly as `run.sh <dir> ltf` starts it, through experiments/alpasim_family/scripts/fam_ltf.py,
# which applies FAM_ARM (fam_perturb.py) to the requests the sample receives.
cwd=$SRC/e2e_challenge/sample_submission_simscale_navsim_transfuser
cmd="exec env TMPDIR=$out/driver-tmp XDG_CACHE_HOME=$out/driver-tmp/cache TORCH_HOME=$out/driver-tmp/torch \
HF_HOME=$out/driver-tmp/hf HF_HUB_OFFLINE=1 MPLCONFIGDIR=$out/driver-tmp/mpl CUDA_CACHE_PATH=$out/driver-tmp/nv \
NUMBA_CACHE_DIR=$out/driver-tmp/numba TORCH_EXTENSIONS_DIR=$out/driver-tmp/torch_ext ALPASIM_DRIVER_LOG_DIR=$out/driver-logs \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1 TORCH_NUM_INTEROP_THREADS=1 \
ALPASIM_DRIVER_GRPC_WORKERS=4 LTF_CHECKPOINT_PATH=$DATA_DIR/models/alpasim_ltf/ltf_sim_navtest.ckpt LTF_DEVICE=cuda \
LTF_MAX_BATCH_SIZE=${LTF_MAX_BATCH_SIZE:-2} LTF_BATCH_WINDOW_MS=2 FAM_ARM='${FAM_ARM:-}' PYTHONPATH=$cwd \
$DATA_DIR/envs/alpasim-ltf/bin/python $here/../../alpasim_family/scripts/fam_ltf.py"
