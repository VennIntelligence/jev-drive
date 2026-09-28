#!/usr/bin/env bash
# Official NAVSIM v1.1 PDMS for every pose file of the op_interp NAVSIM subset that has no score yet
# (research/openpilot-openloop-integration.md). Scores only the subset's tokens (TOKENS_FILE) with the exam's scorer.
#   scripts/op_interp_score.sh [cpus]      e.g. 170-171,174-177,190-199
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
d=$DATA_DIR/runs/op_interp/nav
cpus=${1:-}
export TOKENS_FILE=$d/tokens.txt NAVSIM_THREADS=${NAVSIM_THREADS:-16}
for f in "$d"/preds/*.npz; do
  name=opi_$(basename "$f" .npz)
  ls "$DATA_DIR"/runs/navsim/eval/v1_navtest_"$name"/*/*.csv >/dev/null 2>&1 && continue
  echo "== $name $(date +%H:%M:%S)"
  if [[ -n $cpus ]]; then taskset -c "$cpus" "$repo/scripts/navsim_zs_score.sh" score v1 navtest "$name" "$f" > "$d/score_$name.log" 2>&1
  else "$repo/scripts/navsim_zs_score.sh" score v1 navtest "$name" "$f" > "$d/score_$name.log" 2>&1; fi
  tail -n 3 "$d/score_$name.log"
done
echo "all scored $(date +%H:%M:%S)"
