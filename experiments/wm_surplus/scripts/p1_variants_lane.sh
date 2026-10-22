#!/usr/bin/env bash
# P1 projection-artifact controls on a probed subject (p1_agent.py vgate / vsweep / vmerge; prereg
# plans/2026-10-23-p1-proj-artifact-prereg.md): submits the chain to the GPU pool and returns.
#   vgate (GPU) -> vsweep shards (GPU, one per card: compute-bound, VRAM declared so two never share a card) -> vmerge (CPU)
#   -> readout (CPU: bench navtest scoring of every variant arm, the protocol's arms reused, v1.1 PDMS, paired vs off)
# Usage (box, repo root): experiments/wm_surplus/scripts/p1_variants_lane.sh <model> [<tag>]     e.g. ReCogDrive-2B-RL recog-lmh
# Env: GATE_ONLY=1 submits the gate only; NS sweep shards (default: cards), PRIO (default -1), VRAM (default 45), DIM (1536).
# Wait for $O/variants/results/DONE with one blocking waiter (the readout job writes it), never a polling loop.
set -euo pipefail
: "${DATA_DIR:?}"
cd "$(dirname "$0")/../../.."
M=$1; TAG=${2:-recog-lmh}
O=$DATA_DIR/runs/wm_surplus/p1_agents/$M
V=$O/variants
mkdir -p "$V/pool"
ENV=$(.venv/bin/python -c "
import shlex, sys
from jevdrive.bench.navsim import agent_env
from jevdrive.bench.models import agent_cfg, data_dir
c = agent_cfg(sys.argv[1])
print(' '.join(f'{k}={shlex.quote(v)}' for k, v in agent_env(c).items()), str(data_dir() / 'envs' / c['env'] / 'bin/python'))" "$M")
PY=${ENV##* }; ENV=${ENV% *}
ME=experiments/wm_surplus/scripts/p1_agent.py
cards=$(nvidia-smi -L | wc -l)
NS=${NS:-$cards}; PRIO=${PRIO:--1}; VRAM=${VRAM:-45}
sub() { local n=$1; shift; .venv/bin/python -m jevdrive.cl submit --owner p0-subjects --priority "$PRIO" --log-dir "$V/pool/$n" --name "p1v-$TAG-$n" "$@" | awk '{print $NF}'; }
join() { local IFS=,; echo "$*"; }
if [ -f "$V/gate.json" ] && grep -q '"passed": true' "$V/gate.json"; then G=""; else
  G=$(sub vgate --vram "$VRAM" --cpu 4 --ram 24 -- env $ENV "$PY" $ME vgate --model "$M" --n 24); fi
[ -n "${GATE_ONLY:-}" ] && { echo "vgate $G"; exit 0; }
AFT=(); [ -n "$G" ] && AFT=(--after "$G")
S=(); for i in $(seq 0 $((NS - 1))); do
  S+=($(sub vsweep-s$i "${AFT[@]}" --vram "$VRAM" --cpu 4 --ram 32 -- env $ENV "$PY" $ME vsweep --model "$M" --shard "$i" "$NS"))
done
MG=$(sub vmerge --after "$(join "${S[@]}")" --vram 0.1 --cpu 2 --ram 16 -- env $ENV "$PY" $ME vmerge --model "$M")
ARMS=$(.venv/bin/python - "${DIM:-1536}" "$V/gate.json" <<'PY'
import json, sys
D = int(sys.argv[1]); ks = [2 ** i for i in range(3, 20) if 2 ** i < D] + [D]
km = json.load(open(sys.argv[2]))["kmax"]
arms = ["off"] + [f"k{k}" for k in ks]
for v, m in km.items():
    arms += [f"{v}-k{k}" for k in [k for k in ks if k < m] + ([m] if m <= D else [])]
print(" ".join(arms))
PY
)
RC=32
R=$(sub readout --after "$MG" --vram 0.1 --cpu $RC --ram 32 -- bash -c "env PYTHONPATH=$PWD NAVSIM_THREADS=$((RC - 1)) .venv/bin/python experiments/wm_surplus/scripts/p1_readout.py \
  --tag $TAG --arms $ARMS --ev $V/ev.json --out $V/results --priority $PRIO --v1 && touch $V/results/DONE")
printf 'vgate %s\nvsweep %s\nvmerge %s\nreadout %s\n' "$G" "$(join "${S[@]}")" "$MG" "$R" | tee "$V/jobs.txt"
