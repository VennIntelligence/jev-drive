#!/usr/bin/env bash
# P5 v1 profiling (todos/2026-09-25-reactivity-program/i1-p5v1.md): 2 pairs end to end with both experts, both worlds,
# through the v0 index/labels. Pair A = 27515 seed 0 (PedestrianCrossing, Town03, a v0 route: its BehaviorAgent runs
# double as the v0 reproduction check) with its null; pair B = 11177 seed 0 (VehicleTurningRoutePedestrian, Town12, new
# in v1). CARLA index 880+ on GPU $GPU, pinned to $CPUS, $W instances at a time; the sampler records CPU / RAM / VRAM.
#   scripts/p5v1_profile.sh <tag> [save_threads]
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
tag=$1 st=${2:-0}
CPUS=${PROF_CPUS:-196-207} GPU=${PROF_GPU:-2} W=${PROF_WORKERS:-2}
R=$DATA_DIR/runs/p5v1
P=$R/prof-$tag
mkdir -p "$P"
T=$(python3 -c "import json;print(json.load(open('$R/agent-ba.json'))['tfv6_model_dir'])")
echo "{\"tfv6_model_dir\": \"$T\", \"save_threads\": $st}" > "$P/agent-ba.json"
echo "{\"tfv6_model_dir\": \"$T\", \"save_threads\": $st, \"driver\": \"pdm_lite\"}" > "$P/agent-pdm.json"
export B2D_RESEED_AFTER_BUILD=1 LEAD_PROJECT_ROOT=$DATA_DIR/third_party/scout/lead-cvpr2026 HF_HUB_OFFLINE=1 \
    OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=${NUMBA_THREADS:-3} SAVE_PATH=$R/lead_save
export PYTHONPATH=$LEAD_PROJECT_ROOT
rm -f "$P/stop"
taskset -c "$CPUS" "$DATA_DIR/envs/carla/bin/python" scripts/p5v1_prof_sampler.py --out "$P/prof.tsv" \
    --port-lo 46000 --port-hi 46950 --gpu "$GPU" --until-file "$P/stop" &
t0=$(date +%s)
for e in pdm ba; do
    tree=$DATA_DIR/third_party/Bench2Drive; [[ $e == pdm ]] && tree=$DATA_DIR/third_party/simlingo/Bench2Drive
    CUDA_VISIBLE_DEVICES=$GPU BENCH2DRIVE_ROOT=$tree WORK_DIR=$DATA_DIR/third_party/simlingo taskset -c "$CPUS" \
        "$DATA_DIR/envs/carla/bin/python" scripts/b2d_run.py --routes "$R/pairs.xml" \
        --route-ids 2751510,2751520,2751530,1117710,1117720 --out "$P/gen-$e" --workers "$W" --server-index 880 \
        --index-span 10 --gpu-rank "$GPU" --tm-seed-from-id --agent scripts/p5_pair_agent.py \
        --agent-config "$P/agent-$e.json" --python "$DATA_DIR/envs/scout-tfv6/bin/python" --fast-copy --no-spectator \
        --no-reap --max-attempts 2 --stagger-s 20
    echo "$(date +%T) $e done, wall so far $(( $(date +%s) - t0 )) s"
done
touch "$P/stop"
wait
taskset -c "$CPUS" .venv/bin/python - "$P" <<'EOF'
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
from jevdrive import p5_pairs as Pp, p5v1
P = Path(sys.argv[1])
Pp.RESULTS = p5v1.RESULTS
for e in ("pdm", "ba"):
    t0 = time.time()
    pairs, frames, nulls = Pp.collect(P / f"gen-{e}", only={"27515", "11177"}, workers=4)
    print(f"== {e}: collect {time.time() - t0:.1f} s")
    print(pairs[[c for c in ("base_id", "seed", "t_trig", "t_vis", "t_div", "t_last", "reason", "n_obs", "t_div_null", "n_null")
                 if c in pairs]].query("seed == 0").to_string(index=False))
    if len(frames):
        f = frames[frames.seed == 0]
        print(f.groupby("base_id").d_expert.describe().round(2).to_string())
    for g in (P / f"gen-{e}" / "done").glob("*.json"):
        r = json.loads(g.read_text())
        s = json.loads((P / f"gen-{e}" / "attempts" / r["route_id"] / str(r["attempt"]) / "p5_summary.json").read_text())
        print(r["route_id"], "wall", r["wall_s"], "ticks", s["ticks"], s["stop"], "ms", s["ms_mean"])
# v0 reproduction: BehaviorAgent, v1 code, same worlds as v0
for rid in ("2751510", "2751520"):
    a = Pp.attempt(Path(p5v1.data_dir()) / p5v1.V0_GEN, rid)
    b = Pp.attempt(P / "gen-ba", rid)
    if a is None or b is None:
        print("repro", rid, "missing"); continue
    A, B = Pp.load_world(a), Pp.load_world(b)
    m = A["pose"][["x", "y", "yaw"]].join(B["pose"][["x", "y", "yaw"]], rsuffix="_b", how="inner")
    d = np.hypot(m.x - m.x_b, m.y - m.y_b)
    print("repro", rid, "ticks v0/v1", len(A["pose"]), len(B["pose"]), "common", len(m), "max |dxy| m", round(float(d.max()), 6),
          "first tick >= 1 cm", int(m.index[np.argmax(d.to_numpy() >= 0.01)]) if (d >= 0.01).any() else None)
EOF
.venv/bin/python scripts/p5v1_prof_sampler.py --summary "$P/prof.tsv"
echo "total wall $(( $(date +%s) - t0 )) s"
