#!/usr/bin/env bash
# Official NAVSIM PDMS (v1) / EPDMS (v2) for every pose file of an op_interp NAVSIM run dir that has no score yet
# (research/openpilot-openloop-integration.md). Scores only the run's tokens (TOKENS_FILE) with the exam's scorer.
#   scripts/op_interp_score.sh [cpus] [data] [ver] [split]
#   e.g. scripts/op_interp_score.sh 170-171,174-177,190-199                       # nav, v1, navtest (default: unchanged)
#        scripts/op_interp_score.sh 170-199 navfull v1 navtest                    # full navtest
#        scripts/op_interp_score.sh 170-199 navhard v2 navhard_two_stage          # full navhard two-stage
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
data=${2:-nav}
d=$DATA_DIR/runs/op_interp/$data
ver=${3:-v1} split=${4:-navtest}
cpus=${1:-}
export TOKENS_FILE=$d/tokens.txt NAVSIM_THREADS=${NAVSIM_THREADS:-16}
# name includes the run dir except for the original "nav" (2000-token) diagnostic, kept as-is for backward
# compatibility: pose files of the same stem (e.g. warp-cinque__base) exist in more than one run dir (nav's
# 2000-token subset vs navfull's/navhard's full set) and must not share one eval directory -- they score different
# token sets under the same stem.
[[ $data == nav ]] && pfx=opi_ || pfx=opi_${data}_
for f in "$d"/preds/*.npz; do
  name=${pfx}$(basename "$f" .npz)
  ls "$DATA_DIR"/runs/navsim/eval/${ver}_${split}_"$name"/*/*.csv >/dev/null 2>&1 && continue
  echo "== $name $(date +%H:%M:%S)"
  if [[ -n $cpus ]]; then taskset -c "$cpus" "$repo/scripts/navsim_zs_score.sh" score "$ver" "$split" "$name" "$f" > "$d/score_$name.log" 2>&1
  else "$repo/scripts/navsim_zs_score.sh" score "$ver" "$split" "$name" "$f" > "$d/score_$name.log" 2>&1; fi
  tail -n 3 "$d/score_$name.log"
done
echo "all scored $(date +%H:%M:%S)"
