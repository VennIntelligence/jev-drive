# Sourced by run.sh: the public py123d_garage checkpoint resnet34_v0.1.0/model_0014.pth (camera-only latent TransFuser, ResNet-34) as the
# AlpaSim nuPlan-track driver, run natively in envs/garage (experiments/alpasim_proxy/scripts/setup_garage.sh). Mirrors the garage's
# lib/alpasim/tools/submission.Dockerfile at v0.1.1 (288d34c): its ENV block (write dirs moved off the system disk) and its ENTRYPOINT
# overrides, nuPlan rig sensor_rig_0.yaml next to the checkpoint. The entry point is garage_driver.py, which runs their evaluate.main()
# unchanged and only adds driver-logs/drive.jsonl close records (drive calls, plans, inference / input errors) for driver_health.
GK=$DATA_DIR/models/garage/resnet34_v0.1.0
cwd=$out/driver-tmp
cmd="exec env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HOME=$out/driver-tmp \
TMPDIR=$out/driver-tmp TORCH_HOME=$out/driver-tmp/torch MPLCONFIGDIR=$out/driver-tmp/matplotlib OMP_NUM_THREADS=8 \
ALPASIM_DRIVER_LOG_DIR=$out/driver-logs $DATA_DIR/envs/garage/bin/python $here/../../alpasim_proxy/scripts/garage_driver.py \
hydra.run.dir=$out/driver-tmp/alpasim_driver policy_config.evaluation_checkpoint_file=$GK/model_0014.pth \
policy_config.evaluation_sensor_rig_file=$GK/sensor_rig_0.yaml git_hash=288d34c host=\${ALPASIM_DRIVER_HOST:-0.0.0.0} port=\${ALPASIM_DRIVER_PORT:-6789}"
