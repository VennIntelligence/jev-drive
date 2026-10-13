#!/usr/bin/env bash
# Camera rerun of one review chunk (copied from tmp/held_turns/video_run.sh): bash rf_lateturn_video.sh <TAG> <scene list> <run dir>. = rm_chain.sh sim + eval.video.render_video=true, no prune.
set -euo pipefail
cd "$(dirname "$0")/../../.."
T=$1 L=$2 D=$3
C=$DATA_DIR/datasets/alpasim_nuplan_long/after160; KEY=runtime.scene_provider.trajdata
source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
rm -rf "$D"; mkdir -p "$D"
env SH30_TAG=$T SH30_DUMP=0 bash experiments/alpasim/scripts/run.sh "$D" sh30 --scene-list "$L" "${BASE[@]}" "$KEY.cache_location=$C" \
  "$KEY.dataset.extra_params.num_timesteps_after=160" eval.video.render_video=true
find "$D/rollouts" -name '*.mp4' | wc -l
