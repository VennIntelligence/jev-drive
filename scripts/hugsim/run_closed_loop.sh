#!/usr/bin/env bash
# Run HUGSIM's official closed_loop.py on scenario yamls with our pipe-protocol agent (scripts/hugsim/agent_client.py).
# Usage (on the box): GPU=1 POLICY=route scripts/hugsim/run_closed_loop.sh <out_root> <scenario.yaml> ...
#   dataset is taken from the scenario's parent dir name (nuscenes | waymo | kitti360 | pandaset).
# Writes per scenario <out_root>/<ad>_/<scene>_<mode>/{eval.json,video.mp4,data.pkl,infos.pkl,output.txt}
# plus <out_root>/runs.csv (wall time, steps, peak simulator VRAM from nvidia-smi).
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
D=${DATA_DIR:-$HOME/data}
HUG=$D/third_party/HUGSIM
DATA=$D/datasets/hugsim
PY=$D/envs/hugsim/bin/python
GPU=${GPU:-1}; export HUGSIM_POLICY=${POLICY:-route}
out_root=$(realpath -m "$1"); shift
mkdir -p "$out_root"
[[ -f $out_root/runs.csv ]] || echo "scenario,policy,wall_s,steps,peak_mib,hdscore,rc" > "$out_root/runs.csv"

for scen in "$@"; do
  scen=$(realpath "$scen"); ds=$(basename "$(dirname "$scen")")
  scene=$("$PY" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['scene_name'])" "$scen")
  mode=$("$PY" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['mode'])" "$scen")
  base=$out_root/base_$ds.yaml
  cat > "$base" <<EOF
realcar_path: $DATA/3DRealCar
model_base: $DATA/scenes/$ds
jev_path: $repo/scripts/hugsim/agent_e2e.sh
output_dir: $out_root/
HD_map:
  path: $DATA/nusc_map_cache
  version: nusc_trainval
EOF
  export HUGSIM_SCENE_DIR=$DATA/scenes/$ds/$scene
  run_dir=$out_root/jev/${scene}_$mode
  rm -rf "$run_dir"; mkdir -p "$run_dir"
  echo "== $(date +%H:%M:%S) $ds/$(basename "$scen") policy=$HUGSIM_POLICY gpu=$GPU"
  t0=$SECONDS
  (cd "$HUG" && CUDA_VISIBLE_DEVICES=$GPU "$PY" -u closed_loop.py --scenario_path "$scen" \
      --base_path "$base" --camera_path "configs/sim/${ds}_camera.yaml" \
      --kinematic_path configs/sim/kinematic.yaml --ad jev --ad_cuda "$GPU" > "$run_dir/sim.log" 2>&1) &
  sim=$!
  peak=0
  while kill -0 $sim 2>/dev/null; do  # peak VRAM of the simulator process tree on our GPU
    pids=$(pgrep -P $sim -d'|' || true); pids="$sim${pids:+|$pids}"
    m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i "$GPU" 2>/dev/null \
        | awk -F', ' -v p="^($pids)$" '$1 ~ p {s+=$2} END {print s+0}')
    (( m > peak )) && peak=$m
    sleep 1
  done
  wait $sim || echo "closed_loop.py exited non-zero, see $run_dir/sim.log"
  wall=$((SECONDS - t0))
  steps=$(grep -c "^ego pose" "$run_dir/sim.log" || true)
  read -r hd rc < <("$PY" -c "import json,sys
try: e=json.load(open(sys.argv[1])); print(e['hdscore'], e['rc'])
except Exception: print('nan nan')" "$run_dir/eval.json")
  echo "$ds/$(basename "$scen" .yaml),$HUGSIM_POLICY,$wall,$steps,$peak,$hd,$rc" >> "$out_root/runs.csv"
  echo "   wall ${wall} s, $steps steps, peak ${peak} MiB, HD-Score $hd, RC $rc"
done
