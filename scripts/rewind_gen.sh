#!/usr/bin/env bash
# CARLA rewind fidelity runs (todos/2026-09-29-carla-rewind.md): the planned rewind runs and floor reruns of
# runs/rewind/plan.parquet (scripts/rewind_eval.py prep) under scripts/wl_fork_agent.py, one b2d_run per source set.
#   scripts/tmux_run.sh rewind scripts/rewind_gen.sh     env: IDS=<comma list> (default: every planned run), SETS="ba p6"
#                                                          GEN=gen | gen_reuse (the map-reuse arm: B2D_REUSE_MAP=1, runs
#                                                          ordered by town so same-map routes follow each other)
# Card, workers, server indices and cores come from the `carla-rewind` row of $DATA_DIR/runs/sched/table.tsv.
# Resumable: b2d_run skips routes with a done/<id>.json.
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
R=$DATA_DIR/runs/rewind
GEN=${GEN:-gen}
IFS=$'\t' read -r _ G W IDX SPAN CPUS _ < <(awk -F'\t' '$1 == "carla-rewind"' "$DATA_DIR/runs/sched/table.tsv")
[[ -n ${G:-} ]] || { echo "no carla-rewind row in table.tsv"; exit 1; }
W=${WORKERS:-$W}; SPAN=${IDX_SPAN:-$SPAN}      # fewer servers than the row allows, e.g. to keep index 442 for a probe
python3 scripts/sch_table.py check > /dev/null || { echo "sch_table.py check fails: not starting"; exit 1; }
export B2D_RESEED_AFTER_BUILD=1 B2D_CAPTURE_CRITERION_EVENTS=1 LEAD_PROJECT_ROOT=$DATA_DIR/third_party/scout/lead-cvpr2026 \
    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=3 SAVE_PATH=$R/lead_save B2D_PHASES=1
[[ $GEN == gen_reuse ]] && export B2D_REUSE_MAP=1
export PYTHONPATH=$LEAD_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}
exec > >(tee -a "$R/log.txt") 2>&1
echo "$(date '+%F %T') rewind-gen $GEN: GPU $G x $W, indices $IDX+$SPAN, cores $CPUS, sets ${SETS:-ba p6}, ids ${IDS:-all}"
for s in ${SETS:-ba p6}; do
    ids=$(.venv/bin/python - "$R" "$s" "${IDS:-}" "$GEN" <<'PYEOF'
import os, sys
import xml.etree.ElementTree as ET
import pandas as pd
r, s, want, gen = sys.argv[1], sys.argv[2], set(filter(None, sys.argv[3].split(","))), sys.argv[4]
p = pd.read_parquet(f"{r}/plan.parquet")
p = p[(p.set == s) & (p.gen == gen) & (p.route_id.isin(want) if want else True)].copy()
town = {e.get("id"): e.get("town") for e in ET.parse(f"{r}/routes-{s}.xml").getroot().iter("route")}
p["town"] = p.route_id.map(town)
p = p.sort_values(["town", "fork_id", "route_id"])
print(",".join(x for x in p.route_id if not os.path.exists(f"{r}/{gen}/{s}/done/{x}.json")))
PYEOF
)
    [[ -z $ids ]] && continue
    if [[ $s == ba ]]; then tree=$DATA_DIR/third_party/Bench2Drive; py=$DATA_DIR/envs/scout-tfv6/bin/python
    else tree=$DATA_DIR/third_party/simlingo/Bench2Drive; py=$DATA_DIR/envs/p5v1-pdm/bin/python; fi
    echo "$(date +%T) $s: $(tr ',' '\n' <<< "$ids" | wc -l) runs"
    CUDA_VISIBLE_DEVICES=$G BENCH2DRIVE_ROOT=$tree WORK_DIR=$DATA_DIR/third_party/simlingo taskset -c "$CPUS" \
        "$DATA_DIR/envs/carla/bin/python" scripts/b2d_run.py --routes "$R/routes-$s.xml" --route-ids "$ids" --out "$R/$GEN/$s" \
        --workers "$W" --server-index "$IDX" --index-span "$SPAN" --gpu-rank "$G" --tm-seed-from-id \
        --agent scripts/wl_fork_agent.py --agent-config "$R/agent-$s.json" --python "$py" \
        --fast-copy --no-spectator --no-reap --max-attempts 3 --stagger-s 30 --client-threads 8 --stall-s 600
done
echo "$(date '+%F %T') rewind-gen end"
