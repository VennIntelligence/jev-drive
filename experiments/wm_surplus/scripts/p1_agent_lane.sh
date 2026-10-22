#!/usr/bin/env bash
# P1 level (a) lane on a third-party subject (p1_agent.py): submits the whole chain to the GPU pool at once and returns.
#   fit (2 navtest log folds x NF shards, GPU) -> pca (CPU) -> gate with PCA (GPU: G1 / G3 / G4 bit-identical, full-rank diff)
#   -> sweep shards (GPU) -> merge (CPU) -> readout (CPU: bench navtest scoring of every arm, paired tables vs off, k95 / k99)
#   -> knows-vs-does (CPU, p1_knowdo.py on knowdo_feats.npz and the off plans)
# Usage (box, repo root): experiments/wm_surplus/scripts/p1_agent_lane.sh <model> [<tag>]   e.g. ReCogDrive-2B-RL recog-lmh
# Env: NF fit shards per fold (default 2), NS sweep shards (default: cards), PRIO (default -1), VRAM (default 14).
# Wait for $O/readout/DONE (and $O/knowdo/DONE) with one blocking waiter, never a polling loop.
set -euo pipefail
: "${DATA_DIR:?}"
cd "$(dirname "$0")/../../.."
M=$1; TAG=${2:-recog-lmh}
O=$DATA_DIR/runs/wm_surplus/p1_agents/$M
mkdir -p "$O/pool"
ENV=$(.venv/bin/python -c "
import json, shlex, sys
from jevdrive.bench.navsim import agent_env
from jevdrive.bench.models import agent_cfg, data_dir
c = agent_cfg(sys.argv[1])
print(' '.join(f'{k}={shlex.quote(v)}' for k, v in agent_env(c).items()), str(data_dir() / 'envs' / c['env'] / 'bin/python'))" "$M")
PY=${ENV##* }; ENV=${ENV% *}
ME=experiments/wm_surplus/scripts/p1_agent.py
cards=$(nvidia-smi -L | wc -l)
NF=${NF:-2}; NS=${NS:-$cards}; PRIO=${PRIO:--1}; VRAM=${VRAM:-14}
sub() { local n=$1; shift; .venv/bin/python -m jevdrive.cl submit --owner p0-subjects --priority "$PRIO" --log-dir "$O/pool/$n" --name "p1a-$TAG-$n" "$@" | awk '{print $NF}'; }
join() { local IFS=,; echo "$*"; }
F=(); for f in 0 1; do for i in $(seq 0 $((NF - 1))); do
  F+=($(sub fit-f$f-s$i --vram "$VRAM" --cpu 4 --ram 32 -- env $ENV "$PY" $ME fit --model "$M" --fold $f --shard "$i" "$NF"))
done; done
P=$(sub pca --after "$(join "${F[@]}")" --vram 0.1 --cpu 2 --ram 16 -- env $ENV "$PY" $ME pca --model "$M")
G=$(sub gate-pca --after "$P" --vram "$VRAM" --cpu 4 --ram 24 -- env $ENV "$PY" $ME gate --model "$M" --n 24 --pca)
S=(); for i in $(seq 0 $((NS - 1))); do
  S+=($(sub sweep-s$i --after "$G" --vram "$VRAM" --cpu 4 --ram 32 -- env $ENV "$PY" $ME sweep --model "$M" --shard "$i" "$NS"))
done
MG=$(sub merge --after "$(join "${S[@]}")" --vram 0.1 --cpu 2 --ram 16 -- env $ENV "$PY" $ME merge --model "$M")
ARMS=$(.venv/bin/python -c "
import sys; sys.path.insert(0, 'experiments/wm_surplus/scripts')
D = int(sys.argv[1]); print(' '.join(['off'] + [f'k{2**i}' for i in range(3, 20) if 2**i < D] + [f'k{D}']))" "${DIM:-1536}")
R=$(sub readout --after "$MG" --vram 0.1 --cpu 1 --ram 8 -- env PYTHONPATH="$PWD" .venv/bin/python experiments/wm_surplus/scripts/p1_readout.py \
  --tag "$TAG" --arms $ARMS --ev "$O/ev.json" --out "$O/results" --priority "$PRIO")
K=$(sub knowdo --after "$MG" --vram 0.1 --cpu 8 --ram 16 -- .venv/bin/python experiments/wm_surplus/scripts/p1_knowdo.py \
  --feats "$O/knowdo_feats.npz" --plans "PROBE-$TAG-off" --name "$TAG" --out "$O/knowdo")
printf 'fit %s\npca %s\ngate %s\nsweep %s\nmerge %s\nreadout %s\nknowdo %s\n' "$(join "${F[@]}")" "$P" "$G" "$(join "${S[@]}")" "$MG" "$R" "$K" | tee "$O/jobs.txt"
