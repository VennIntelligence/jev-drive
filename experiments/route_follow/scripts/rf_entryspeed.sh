#!/usr/bin/env bash
# ENTRY-SPEED ROWS (plans/2026-10-14-entry-speed-rows-prereg.md; rf_entryspeed.py docstring). CPU only, outside the GPU pool; check
# `python -m jevdrive.cl top` for the CPU and memory margin first.
#   box     rf_entryspeed.sh follow     rf_follow on the 88 turn scenes (turn_scenes.txt: the union of the stored reads' turns) of the four row pools without a stored read (tmux, 4 workers: an
#                                       rf_follow worker grows to ~18 GB over ~500 rollouts, 8 workers over all scenes hit the kill line)
#           rf_entryspeed.sh nup        the per-pool extracts -> $DATA_DIR/runs/route_follow/entry_speed/nup
#   Tokyo   rf_entryspeed.sh p3         the P3 extract (plain python3) -> /data/runs/route_night/entry_speed/p3.pkl
#   Mac     rf_entryspeed.sh report     copy both extracts into tmp/ first (scp), then tables / json / figures
set -euo pipefail; cd "$(dirname "$0")/../../.."
X=${ENTRY_PY:-experiments/route_follow/scripts/rf_entryspeed.py}
case $1 in
follow)
  F=$DATA_DIR/runs/route_follow; R=$DATA_DIR/runs/onpol_long/roll; lab(){ echo "$1=$(ls -d $R/$2/c*/run | paste -sd,)"; }
  scripts/tmux_run.sh es-follow $DATA_DIR/third_party/alpasim/.venv/bin/python experiments/route_follow/scripts/rf_follow.py \
    --runs "$(lab RSG-L18-s0@18 RSG-L18-s0)" "$(lab RBSG-L18-s0@18 RBSG-L18-s0)" "$(lab RBSG2-L18-s0@18 RBSG2-L18-s0)" \
           "$(lab RSG-L18-s0@held RSG-L18-s0-held)" "$(lab RBSG2-L18-s0@held RBSG2-L18-s0-held)" \
    --scenes $F/entry_speed/turn_scenes.txt --out $F/entry_speed/follow2 --workers 4 ;;
nup)
  F=$DATA_DIR/runs/route_follow
  .venv/bin/python $X extract-nup --out $F/entry_speed/nup --pool L18RB0=RB0-F-s0@18=$F/report/follow_train_rb \
    L18RSG=RSG-L18-s0@18=$F/entry_speed/follow2 L18RBSG=RBSG-L18-s0@18=$F/entry_speed/follow2 L18RBSG2=RBSG2-L18-s0@18=$F/entry_speed/follow2 \
    L18RSGallheld=RSG-L18-s0@held=$F/entry_speed/follow2 L18RBSG2allheld=RBSG2-L18-s0@held=$F/entry_speed/follow2 ;;
p3)
  python3 $X extract-p3 --arms /data/runs/route_night/p3/arms --out /data/runs/route_night/entry_speed/p3.pkl ;;
report)
  .venv/bin/python $X report --nup tmp/es_nup --p3 tmp/es_p3.pkl --read experiments/route_follow/results/p3_branch/read.json \
    --plan experiments/route_follow/results/p4_self/pool_dev/plan_p3.json --out experiments/route_follow/results/entry_speed_rows \
    --figs experiments/route_follow/figs ;;
esac
