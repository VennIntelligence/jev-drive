#!/usr/bin/env bash
# One PAI-track closed-loop run on the GPU box, no containers: the wizard writes the dev preset (as pai_run.sh does on the Tokyo box), run_native.py
# starts every service as a host process; the renderer is the unpacked nre-ga image (pai_nre_pull.sh), the dev preset's `--enable-harmonizer`
# is dropped (the leaderboard's renderer), the renderer's /tmp cache is per run. Submit it to the pool:
#   python -m jevdrive.cl submit --name pai-<tag> --vram 24 --cpu 12 --log-dir <run dir>/pool -- \
#     bash experiments/alpasim/scripts/pai_native.sh <run dir> <scenes.tsv | file of scene ids> [wizard overrides...]
# Env: CONC (concurrent rollouts, default 4), STAGE (rows of the tsv with stage <= this, default 1), SH30_TAG (checkpoint, default P2H10-F-s0),
#   NUREC (scene cache, default $DATA_DIR/datasets/nurec), NRE_ROOT (default $DATA_DIR/tools/nre-rootfs), JEV_VCONT / JEV_LEAD / PAI_* (drivers/pai.sh).
# The pool sets CUDA_VISIBLE_DEVICES: the card is index 0 inside the run.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); source "$here/env.sh"
OUT=$1 LIST=$2; shift 2
SRC=${ALPASIM_SRC:-$DATA_DIR/third_party/alpasim}; NUREC=${NUREC:-$DATA_DIR/datasets/nurec}; ROOT=${NRE_ROOT:-$DATA_DIR/tools/nre-rootfs}; CONC=${CONC:-4}
mkdir -p "$OUT"
if head -1 "$LIST" | grep -q '^scene_id'; then ids=$(awk -F'\t' -v s="${STAGE:-1}" 'NR > 1 && $8 <= s {print $1}' "$LIST"); else ids=$(grep . "$LIST"); fi
echo "$ids" > "$OUT/scene_ids.txt"
exec bash "$here/run.sh" "$OUT" pai --scene-list "$OUT/scene_ids.txt" --rewrite-configs \
  --sub "/app=$ROOT/app" --sub " --enable-harmonizer=" --sub "/tmp/nre-cache-dir=$OUT/nre-cache" \
  +e2e_challenge=dev scenes.scene_cache="$NUREC" "scenes.scenes_csv=[$SRC/data/scenes/sim_scenes.csv,$SRC/data/scenes/sim_scenes_2604.csv]" \
  "services.renderer.gpus=[0]" "services.physics.gpus=[0]" \
  "runtime.endpoints.renderer.n_concurrent_rollouts=$CONC" "runtime.endpoints.driver.n_concurrent_rollouts=$CONC" \
  "runtime.endpoints.physics.n_concurrent_rollouts=$CONC" "runtime.endpoints.controller.n_concurrent_rollouts=$CONC" \
  "defines.nre_cache_size=$((CONC + 1))" "$@"
