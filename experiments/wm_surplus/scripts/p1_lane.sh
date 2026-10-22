#!/usr/bin/env bash
# P1 level (a) lane on WA-JEPA: submits the whole chain to the GPU pool at once and returns.
#   fit shards (GPU, wait for the navtrain download marker) -> pca (CPU) -> sweep shards (GPU, after pca + identity gate)
#   -> merge (CPU) -> readout (CPU: bench navtest scoring of every arm, sharded over the box's cores; paired tables; k95 / k99)
# Usage (on the box, repo root): GATE=<gate job id> experiments/wm_surplus/scripts/p1_lane.sh
# Env: NF fit shards (default: number of cards), NS sweep shards (default: number of cards), PRIO (default 5).
# Wait for $O/readout/DONE or ERROR with one blocking waiter (no polling).
set -euo pipefail
: "${DATA_DIR:?}"
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/wm_surplus/p1_usage
PY=$DATA_DIR/envs/wajepa/bin/python
ME=experiments/wm_surplus/scripts/p1_wajepa.py
cards=$(nvidia-smi -L | wc -l)
NF=${NF:-$cards}; NS=${NS:-$cards}; PRIO=${PRIO:-5}
sub() { local n=$1; shift; .venv/bin/python -m jevdrive.cl submit --owner p1-usage --priority "$PRIO" --log-dir "$O/pool/$n" --name "p1-$n" "$@" | awk '{print $NF}'; }
join() { local IFS=,; echo "$*"; }
F=(); for i in $(seq 0 $((NF - 1))); do
  F+=($(sub fit-s$i --when-exists "$O/dl/ALL_DONE" --vram 24 --cpu 14 --ram 48 -- "$PY" $ME fit --shard "$i" "$NF"))
done
P=$(sub pca --after "$(join "${F[@]}")" --vram 0.1 --cpu 2 --ram 8 -- "$PY" $ME pca)
after=$P; [[ -n ${GATE:-} ]] && after="$P,$GATE"
S=(); for i in $(seq 0 $((NS - 1))); do
  S+=($(sub sweep-s$i --after "$after" --vram 24 --cpu 14 --ram 48 -- "$PY" $ME sweep --shard "$i" "$NS"))
done
M=$(sub merge --after "$(join "${S[@]}")" --vram 0.1 --cpu 2 --ram 16 -- "$PY" $ME merge)
R=$(sub readout --after "$M" --vram 0.1 --cpu 1 --ram 8 -- .venv/bin/python experiments/wm_surplus/scripts/p1_readout.py \
  --tag wajepa-ctx --arms off k8 k16 k32 k64 k128 k256 k512 --ev "$O/ev.json" --out "$O/results" --priority "$PRIO")
printf 'fit %s\npca %s\nsweep %s\nmerge %s\nreadout %s\n' "$(join "${F[@]}")" "$P" "$(join "${S[@]}")" "$M" "$R" | tee "$O/jobs.txt"
