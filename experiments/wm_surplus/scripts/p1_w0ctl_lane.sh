#!/usr/bin/env bash
# DriveVLA-W0 projection controls (prereg addendum A, plans/2026-10-23-p1-proj-artifact-prereg.md; p1_agent.py wcgate / wcsweep
# / wcmerge) on the fixed subset navsim/p1-w0-ctl: submits the chain to the GPU pool and returns.
#   wcgate (GPU, 24 tokens) -> wcsweep shards (GPU, one per card, VRAM declared so two never share a card) -> wcmerge (CPU)
#   -> readout (CPU: subset bench scoring of the control arms and the original k64 / k512 / k2048 arms, v1.1 PDMS, paired vs off)
# Usage (box, repo root): experiments/wm_surplus/scripts/p1_w0ctl_lane.sh      Env: NS shards (2), PRIO (-1), VRAM (45), SCPU (8).
# Wait for $O/controls/results/DONE with one blocking waiter (the readout job writes it), never a polling loop.
set -euo pipefail
: "${DATA_DIR:?}"
cd "$(dirname "$0")/../../.."
M=DriveVLA-W0; TAG=w0-vlmkv
V=$DATA_DIR/runs/wm_surplus/p1_agents/$M/controls
mkdir -p "$V/pool"
ENV=$(.venv/bin/python -c "
import shlex, sys
from jevdrive.bench.navsim import agent_env
from jevdrive.bench.models import agent_cfg, data_dir
c = agent_cfg(sys.argv[1])
print(' '.join(f'{k}={shlex.quote(v)}' for k, v in agent_env(c).items()), str(data_dir() / 'envs' / c['env'] / 'bin/python'))" "$M")
PY=${ENV##* }; ENV=${ENV% *}
ME=experiments/wm_surplus/scripts/p1_agent.py
NS=${NS:-2}; PRIO=${PRIO:--1}; VRAM=${VRAM:-45}; SCPU=${SCPU:-8}
sub() { local n=$1; shift; .venv/bin/python -m jevdrive.cl submit --owner p0-subjects --priority "$PRIO" --log-dir "$V/pool/$n" --name "p1w0c-$n" "$@" | awk '{print $NF}'; }
join() { local IFS=,; echo "$*"; }
G=$(sub wcgate --vram "$VRAM" --cpu "$SCPU" --ram 48 -- env $ENV "$PY" $ME wcgate --model "$M" --n 24)
S=(); for i in $(seq 0 $((NS - 1))); do
  S+=($(sub wcsweep-s$i --after "$G" --vram "$VRAM" --cpu "$SCPU" --ram 48 -- env $ENV "$PY" $ME wcsweep --model "$M" --shard "$i" "$NS"))
done
MG=$(sub wcmerge --after "$(join "${S[@]}")" --vram 0.1 --cpu 2 --ram 16 -- env $ENV "$PY" $ME wcmerge --model "$M")
ARMS="off ctl-off k64 k512 k2048 norm-k64 norm-k512 norm-k2048 L16-k64 L16-k512 L31-k64 L31-k512"
RC=32
R=$(sub readout --after "$MG" --vram 0.1 --cpu $RC --ram 32 -- bash -c "env PYTHONPATH=$PWD NAVSIM_THREADS=$((RC - 1)) .venv/bin/python experiments/wm_surplus/scripts/p1_readout.py \
  --tag $TAG --arms $ARMS --subset navsim/p1-w0-ctl --out $V/results --priority $PRIO --v1 && touch $V/results/DONE")
printf 'wcgate %s\nwcsweep %s\nwcmerge %s\nreadout %s\n' "$G" "$(join "${S[@]}")" "$MG" "$R" | tee "$V/jobs.txt"
