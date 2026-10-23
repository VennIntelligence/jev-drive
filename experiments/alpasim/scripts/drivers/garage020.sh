# Sourced by run.sh: the host's reference baseline py123d_garage v0.2.0 (ltf-base-joint64-68M, default posttrain/model_0.pth = the
# recommended seed, conditional: target points 40 / 60 / 80 m) as the AlpaSim PAI driver, run natively in envs/garage020
# (setup_garage020.sh). Mirrors their submission.Dockerfile at v0.2.0 (c5db115): its ENV block (write dirs moved off the system disk) and
# ENTRYPOINT (python -m py123d_garage.evaluation.alpasim.evaluate + PY123D_GARAGE_CONFIG), rig sensor_rig_nurec.yaml as their
# scripts/evaluation/alpasim_pai.sh picks for PAI. Driver defaults as shipped: replan every 500 ms, no route memory / extrapolation, no hood.
# The entry point is alpasim_proxy/scripts/garage_driver.py: it runs their evaluate.main() unchanged and only adds driver-logs/drive.jsonl
# close records (drive calls, plans, input errors) for driver_health.
# Env: GARAGE_CKPT (checkpoint, default posttrain/model_0.pth under $GK = conditional; midtrain/model_0.pth = their unconditional stage;
#   the rig is the sensor_rig_nurec.yaml next to it, as alpasim_pai.sh picks), GARAGE_OVERRIDES (their DRIVER_OVERRIDES, space separated, default none).
GK=$DATA_DIR/checkpoints/py123d_garage/v0.2.0/ltf-base-joint64-68M
cwd=$out/driver-tmp
cmd="exec env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HOME=$out/driver-tmp \
TMPDIR=$out/driver-tmp TORCH_HOME=$out/driver-tmp/torch MPLCONFIGDIR=$out/driver-tmp/matplotlib OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
POLARS_MAX_THREADS=8 RAYON_NUM_THREADS=8 OPENCV_FOR_THREADS_NUM=8 \
ALPASIM_DRIVER_LOG_DIR=$out/driver-logs $DATA_DIR/envs/garage020/bin/python $here/../../alpasim_proxy/scripts/garage_driver.py \
hydra.run.dir=$out/driver-tmp/alpasim_driver 'policy_config.evaluation_checkpoint_files=[${GARAGE_CKPT:-$GK/posttrain/model_0.pth}]' \
policy_config.evaluation_sensor_rig_file=$(dirname ${GARAGE_CKPT:-$GK/posttrain/model_0.pth})/sensor_rig_nurec.yaml git_hash=c5db115 ${GARAGE_OVERRIDES:-} \
host=\${ALPASIM_DRIVER_HOST:-0.0.0.0} port=\${ALPASIM_DRIVER_PORT:-6789}"
