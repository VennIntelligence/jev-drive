#!/usr/bin/env bash
# LOWDIAG2: extraction + forced-command replay + unit table for one base seed of the served PAI stacks (frozen scripts of decision 237).
#   scripts/tmux_run.sh lbd2-post-s1 bash experiments/lowboard_diag/scripts/lbd2_pai_post.sh 1 $DATA_DIR/runs/alpasim/fix1/paibox/runs
# Root: $DATA_DIR/runs/lowboard_diag/pai_s<seed>/{x, replay, out}. The replay is one pool job (GPU); extraction and the table run in the pool job too.
set -euo pipefail
cd "$(dirname "$0")/../../.."
SEED=${1:?seed}; STACKS=${2:?dir holding ab_<chunk>_s<seed>}; REL=runs/lowboard_diag/pai_s$SEED; R=$DATA_DIR/$REL; SRC=$DATA_DIR/third_party/alpasim
mkdir -p "$R/pool"; HERE=$PWD
JOB="set -e; cd $SRC; PYTHONPATH=src/utils .venv/bin/python $HERE/experiments/lowboard_diag/scripts/lbd_pai_x.py --stacks $STACKS/ab_*_s$SEED --out $R/x --jobs 6; cd $HERE;
 env ALPASIM_SRC=$SRC SH30_TAG=P2H10-F-s$SEED $DATA_DIR/envs/op-train/bin/python experiments/lowboard_diag/scripts/lbd_pai_replay.py --tag P2H10-F-s$SEED --msgs \$(ls -d $R/x/ab_*_s$SEED/msgs) --out $R/replay;
 LBD_PAI_ROOT=$REL .venv/bin/python experiments/lowboard_diag/scripts/lbd_pai.py units --out $R/out"
.venv/bin/python -m jevdrive.cl submit --no-check --owner lowdiag2 --name "lbd2-pai-post-s$SEED" --vram 24 --cpu 8 --ram 60 --timeout-h 3 --tries 1 --log-dir "$R/pool" -- bash -c "$JOB"
