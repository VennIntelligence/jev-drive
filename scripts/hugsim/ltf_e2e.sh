#!/usr/bin/env bash
# LTF client launcher in the shape closed_loop.py expects (`zsh <this> <cuda_id> <output_dir>`); replaces the
# fork's ltf_e2e.sh, which hard-codes the authors' paths and `pixi run`. Env: scripts/hugsim/install_ltf.sh.
D=${DATA_DIR:-$HOME/data}
cd "$D/third_party/hugsim_deps/NAVSIM" || exit 1
CUDA_VISIBLE_DEVICES=$1 exec "$D/envs/hugsim-ltf/bin/python" -u ltf_e2e.py output="$2"
