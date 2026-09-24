#!/usr/bin/env bash
# Run HUGSIM's official closed_loop.py on scenario yamls with our pipe-protocol agent (scripts/hugsim/agent_client.py).
# Usage (on the box): GPU=1 [AD=jev POLICY=route | AD=ltf] scripts/hugsim/run_closed_loop.sh <out_root> <scenario.yaml> ...
#   dataset is taken from the scenario's parent dir name (nuscenes | waymo | kitti360 | pandaset).
# Writes per scenario <out_root>/<ad>/<scene>_<mode>/{eval.json,video.mp4,data.pkl,infos.pkl,output.txt}
# plus <out_root>/runs.csv (wall time, steps, peak simulator VRAM from nvidia-smi).
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
D=${DATA_DIR:-$HOME/data}
HUG=$D/third_party/HUGSIM
DATA=$D/datasets/hugsim
PY=$D/envs/hugsim/bin/python
GPU=${GPU:-1}; AD=${AD:-jev}; export HUGSIM_POLICY=${POLICY:-route}
[[ $AD == jev ]] || HUGSIM_POLICY=$AD
out_root=$(realpath -m "$1"); shift
mkdir -p "$out_root"
[[ -f $out_root/runs.csv ]] || echo "scenario,policy,wall_s,steps,sim_peak_mib,sim_plus_agent_peak_mib,hdscore,rc" > "$out_root/runs.csv"

for scen in "$@"; do
  scen=$(realpath "$scen"); ds=$(basename "$(dirname "$scen")")
  scene=$("$PY" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['scene_name'])" "$scen")
  mode=$("$PY" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['mode'])" "$scen")
  base=$out_root/base_$ds.yaml
  cat > "$base" <<EOF
realcar_path: $DATA/3DRealCar
model_base: $DATA/scenes/$ds
jev_path: $repo/scripts/hugsim/agent_e2e.sh
ltf_path: $repo/scripts/hugsim/ltf_e2e.sh
output_dir: $out_root/
HD_map:
  path: $DATA/nusc_map_cache
  version: nusc_trainval
EOF
  export HUGSIM_SCENE_DIR=$DATA/scenes/$ds/$scene
  if [[ ! -f $HUGSIM_SCENE_DIR/scene.pth ]]; then  # scenes ship as <scene>.zip; unpack once, next to the zip
    (cd "$DATA/scenes/$ds" && "$PY" -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall('.')" "$scene.zip")
  fi
  run_dir=$out_root/$AD/${scene}_$mode
  rm -rf "$run_dir"; mkdir -p "$run_dir"
  echo "== $(date +%H:%M:%S) $ds/$(basename "$scen") policy=$HUGSIM_POLICY gpu=$GPU"
  t0=$SECONDS
  (cd "$HUG" && CUDA_VISIBLE_DEVICES=$GPU "$PY" -u closed_loop.py --scenario_path "$scen" \
      --base_path "$base" --camera_path "configs/sim/${ds}_camera.yaml" \
      --kinematic_path configs/sim/kinematic.yaml --ad "$AD" --ad_cuda "$GPU" > "$run_dir/sim.log" 2>&1) &
  sim=$!
  peak=0; peak_all=0
  while kill -0 $sim 2>/dev/null; do  # peak VRAM on our GPU: simulator python alone, and with its agent tree
    tree=$(ps -eo pid=,ppid= | awk -v r=$sim '{p[$1]=$2} END {for (i in p) {j=i; while (j in p && j!=r) j=p[j]; if (j==r) print i}}')
    simpy=$(pgrep -P $sim -f closed_loop.py | head -1 || true)
    read -r m m_all < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i "$GPU" 2>/dev/null \
        | awk -F', ' -v s="${simpy:-x}" -v t="$(echo $tree)" 'BEGIN {n=split(t,a," "); for (i=1;i<=n;i++) T[a[i]]=1}
            $1==s {m+=$2} ($1 in T) {ma+=$2} END {print m+0, ma+0}')
    (( m > peak )) && peak=$m; (( m_all > peak_all )) && peak_all=$m_all
    sleep 1
  done
  wait $sim || echo "closed_loop.py exited non-zero, see $run_dir/sim.log"
  wall=$((SECONDS - t0))
  steps=$(grep -c "^ego pose" "$run_dir/sim.log" || true)
  read -r hd rc < <("$PY" -c "import json,sys
try: e=json.load(open(sys.argv[1])); print(e['hdscore'], e['rc'])
except Exception: print('nan nan')" "$run_dir/eval.json")
  echo "$ds/$(basename "$scen" .yaml),$HUGSIM_POLICY,$wall,$steps,$peak,$peak_all,$hd,$rc" >> "$out_root/runs.csv"
  echo "   wall ${wall} s, $steps steps, sim peak ${peak} MiB (with agent ${peak_all}), HD-Score $hd, RC $rc"
done
