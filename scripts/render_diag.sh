#!/usr/bin/env bash
# CARLA render-failure diagnosis (todos/2026-09-28-wm-loop.md, "渲染故障的根因"): re-run WL fork routes with every
# camera frame saved from tick 1, REPS independent repetitions in parallel on one card.
#   scripts/tmux_run.sh rf-<exp> scripts/render_diag.sh
#   env: EXP=name ROUTES=id,id,.. [SET=ba|p6] [REPS=3] [WORKERS=2] [GPU=6] [BASE=460] [CPUS=144-167] [TFV6=0|1]
#        [CAM_ATTRS='{"exposure_mode": "manual"}'] [LIGHTS_TRUTH=40] [LIGHTS_FIX=0|1] [QUALITY=Epic|Low]
#        [RECYCLE=n: restart each server every n routes; 1 = every route on a fresh server]
# Out: $DATA_DIR/runs/wl/renderfix/<exp>/rep<r>/ (b2d_run layout). Rep r uses server indices BASE + 2*WORKERS*r ..
# (+ 2*WORKERS); keep REPS * WORKERS <= 6 per card. Resumable per rep (done/<id>.json).
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}" "${EXP:?EXP}" "${ROUTES:?ROUTES}"
cd "$(dirname "$0")/.."
R=$DATA_DIR/runs/wl
S=${SET:-ba} REPS=${REPS:-3} W=${WORKERS:-2} GPU=${GPU:-6} BASE=${BASE:-460}
OUT=$R/renderfix/$EXP
mkdir -p "$OUT"
exec > >(tee -a "$OUT/log.txt") 2>&1
[[ $S == ba ]] && TREE=$DATA_DIR/third_party/Bench2Drive PYX=$DATA_DIR/envs/scout-tfv6/bin/python SRC=$DATA_DIR/runs/p5v1/agent-ba.json \
               || TREE=$DATA_DIR/third_party/simlingo/Bench2Drive PYX=$DATA_DIR/envs/p5v1-pdm/bin/python SRC=$DATA_DIR/runs/p6/agent-p6.json
python3 - "$SRC" "$OUT/agent.json" "$R/jobs.json" "${TFV6:-0}" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(wl_jobs=sys.argv[3], wl_pre_cams=100000, wl_norender=False)   # save every camera frame from tick 1
if sys.argv[4] != "1":
    c["tfv6_model_dir"] = ""
json.dump(c, open(sys.argv[2], "w"))
PYEOF
export B2D_RESEED_AFTER_BUILD=1 B2D_CAPTURE_CRITERION_EVENTS=1 LEAD_PROJECT_ROOT=$DATA_DIR/third_party/scout/lead-cvpr2026 \
    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=3 SAVE_PATH=$R/lead_save B2D_LIGHTS_TRUTH=${LIGHTS_TRUTH:-40}
export PYTHONPATH=$LEAD_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}
[[ -n ${CAM_ATTRS:-} ]] && export B2D_CAM_ATTRS=$CAM_ATTRS
[[ -n ${LIGHTS_FIX:-} ]] && export B2D_LIGHTS_FIX=$LIGHTS_FIX
echo "$(date '+%F %T') render_diag $EXP: set $S, $REPS reps x $W workers on GPU $GPU, routes $ROUTES, TFV6=${TFV6:-0}" \
     "CAM_ATTRS=${CAM_ATTRS:-} LIGHTS_FIX=${LIGHTS_FIX:-} QUALITY=${QUALITY:-Epic} RECYCLE=${RECYCLE:-0}"
pids=()
for ((r = 0; r < REPS; r++)); do
    CUDA_VISIBLE_DEVICES=$GPU BENCH2DRIVE_ROOT=$TREE WORK_DIR=$DATA_DIR/third_party/simlingo taskset -c "${CPUS:-144-167}" \
        "$DATA_DIR/envs/carla/bin/python" scripts/b2d_run.py --routes "$R/forks-$S.xml" --route-ids "$ROUTES" \
        --out "$OUT/rep$r" --workers "$W" --server-index $(( BASE + 2 * W * r )) --index-span $(( 2 * W )) --gpu-rank "$GPU" \
        --tm-seed-from-id --agent scripts/wl_fork_agent.py --agent-config "$OUT/agent.json" --python "$PYX" \
        --quality "${QUALITY:-Epic}" --fast-copy --no-spectator --no-reap --max-attempts 3 --stagger-s 20 --client-threads 8 \
        --stall-s 600 --recycle-routes ${RECYCLE:-0} > "$OUT/runner-$r.log" 2>&1 &
    pids+=($!)
    echo "runner $! rep $r" >> "$OUT/pids.txt"
    sleep 10
done
for p in "${pids[@]}"; do wait "$p"; done
echo "$(date '+%F %T') render_diag $EXP end"
touch "$OUT/DONE"
