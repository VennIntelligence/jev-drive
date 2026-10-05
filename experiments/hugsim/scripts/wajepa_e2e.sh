#!/usr/bin/env bash
# WA-JEPA client launcher in the shape closed_loop.py expects (`zsh <this> <cuda_id> <output_dir>`, ad slot `wj`).
# Runs the shipped client unchanged (third_party/wajepa close_loop/hugsim_client.py at bec2966) through its own launcher
# scripts/evaluation/run_hugsim_ad.sh (preflight + exec). Env: third_party/wajepa/jev_install.sh -> $DATA_DIR/envs/wajepa.
# Set by experiments/hugsim/archive/zs_run.py: HUGSIM_ZS_CAMYAML. Optional: WM_SAVE_VIZ (default 1: per-step inputs and plans).
D=${DATA_DIR:-$HOME/data}
export WM_REPO=$D/third_party/wajepa
export WM_PYTHON=$D/envs/wajepa/bin/python
export WM_CONFIG=$WM_REPO/configs/wa_jepa_hugsim.yaml
export WM_CHECKPOINT=$D/models/wajepa
export WM_SAVE_VIZ=${WM_SAVE_VIZ:-1}
export CAMERA_YAML=${HUGSIM_ZS_CAMYAML:-}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
exec bash "$WM_REPO/scripts/evaluation/run_hugsim_ad.sh" "$1" "$2"
