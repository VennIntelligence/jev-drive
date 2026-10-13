# Shared environment of the monitor launchers (sourced): repo root, monitor dir, CARLA / Bench2Drive roots, card, the desktop display.
# Monitor dir $B2D_MON_DIR (default $DATA_DIR/runs/b2d_monitor): live/ (preview + picker files), srv/ (sockets, server logs), runs/
# (loop output), carla-logs/, cfg-resolved/. Card: MON_GPU (default 0). DISPLAY defaults to :0; XAUTHORITY, unless set and readable,
# is the file the running Xwayland got with -auth (docs/tokyo-box.md, "Looking at CARLA").
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
: "${DATA_DIR:?DATA_DIR is not set, see docs/storage.md}"
export DATA_DIR B2D_MON_DIR=${B2D_MON_DIR:-$DATA_DIR/runs/b2d_monitor} MON_GPU=${MON_GPU:-0}
export CARLA_ROOT=${CARLA_ROOT:-$DATA_DIR/third_party/carla/CARLA_0.9.15} BENCH2DRIVE_ROOT=${BENCH2DRIVE_ROOT:-$DATA_DIR/third_party/Bench2Drive}
export DISPLAY=${DISPLAY:-:0}
[[ -n ${XAUTHORITY:-} && -r $XAUTHORITY ]] || export XAUTHORITY="$(ps -C Xwayland -o args= | sed -n 's/.*-auth \([^ ]*\).*/\1/p' | head -1)"
V=$B2D_MON_DIR
PY=$DATA_DIR/envs/carla/bin/python
XML=$BENCH2DRIVE_ROOT/leaderboard/data/bench2drive_0.0.4_val.xml
MON_IDS=24721,24944,27005,24416,27870,24519,26023,334,24948,28147,27297     # the 11-route mix (README)
mkdir -p "$V/live" "$V/srv" "$V/runs"
# route-process env of the closed-loop runs (one CARLA, one route process)
B2D_RUN_ENV="B2D_PIDS_WAIT=17000 B2D_SENSOR_TICK=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1"
cd "$REPO"
