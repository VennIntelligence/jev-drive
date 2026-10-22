#!/usr/bin/env bash
# SimScale LTF / DiffusionDrive HUGSIM client launcher in the shape closed_loop.py expects (`zsh <this> <cuda_id> <output_dir>`,
# ad slot `ss`). The model comes from HUGSIM_ZS_MODEL (ss-ltf | ss-dd), set by experiments/hugsim/archive/zs_run.py.
# Client: experiments/hugsim/scripts/simscale_hugsim_client.py; env envs/alpasim-ltf.
D=${DATA_DIR:-$HOME/data}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
CUDA_VISIBLE_DEVICES=$1 exec "$D/envs/alpasim-ltf/bin/python" -u "$(dirname "$0")/simscale_hugsim_client.py" \
    --model "${HUGSIM_ZS_MODEL:?HUGSIM_ZS_MODEL must be ss-ltf or ss-dd}" --output "$2"
