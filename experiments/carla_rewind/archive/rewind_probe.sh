#!/usr/bin/env bash
# One CARLA server on the carla-rewind row's last index for experiments/carla_rewind/archive/rewind_physics_probe.py (fc65452:todos/2026-09-29-carla-rewind.md).
#   scripts/tmux_run.sh rw-probe experiments/carla_rewind/archive/rewind_probe.sh [probe args]
set -uo pipefail
: "${DATA_DIR:?}"
cd "$(dirname "$0")/../../.."
IFS=$'\t' read -r _ G W IDX SPAN CPUS _ < <(awk -F'\t' '$1 == "carla-rewind"' "$DATA_DIR/runs/sched/table.tsv")
i=$(( IDX + SPAN - 1 )); port=$(( 2000 + 50 * i )); tm=$(( 8000 + 50 * i ))
out=$DATA_DIR/runs/rewind/probe; mkdir -p "$out"
icd=/etc/vulkan/icd.d/nvidia_icd.json; [[ -e $icd ]] || icd=/usr/share/vulkan/icd.d/nvidia_icd.json
VK_ICD_FILENAMES=$icd setsid taskset -c "$CPUS" "$DATA_DIR/third_party/carla/CARLA_0.9.15/CarlaUE4.sh" -RenderOffScreen -nosound \
    -carla-rpc-port=$port -graphicsadapter=$G -RPCThreads=4 -StreamingThreads=4 -SecondaryThreads=4 > "$out/carla-$i.log" 2>&1 &
spid=$!
trap 'kill -TERM -- -$spid 2>/dev/null; sleep 3; kill -KILL -- -$spid 2>/dev/null' EXIT
for _ in $(seq 90); do (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && break; sleep 2; done
echo "server $i (pgid $spid) on $port"
taskset -c "$CPUS" "$DATA_DIR/envs/carla/bin/python" experiments/carla_rewind/archive/rewind_physics_probe.py --port $port --tm-port $tm \
    --out "$out/probe_$(date +%H%M%S).json" "$@"
