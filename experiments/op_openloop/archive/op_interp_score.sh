#!/usr/bin/env bash
# Official NAVSIM PDMS (v1) / EPDMS (v2) for every pose file of an op_interp NAVSIM run dir that has no score yet
# (research/openpilot-openloop-integration.md). Scores only the run's tokens (TOKENS_FILE) with the exam's scorer.
#   experiments/op_openloop/archive/op_interp_score.sh [cpus] [data] [ver] [split]
#   e.g. experiments/op_openloop/archive/op_interp_score.sh 170-171,174-177,190-199                       # nav, v1, navtest (default: unchanged)
#        experiments/op_openloop/archive/op_interp_score.sh 170-199 navfull v1 navtest                    # full navtest
#        experiments/op_openloop/archive/op_interp_score.sh 170-199 navhard v2 navhard_two_stage          # full navhard two-stage
set -uo pipefail
repo=$(cd "$(dirname "$0")/../../.." && pwd)
data=${2:-nav}
d=$DATA_DIR/runs/${OPI_ROOT:-op_interp}/$data   # OPI_ROOT=op_lb: scripts/op_lb.py run dirs
ver=${3:-v1} split=${4:-navtest}
cpus=${1:-}
export NAVSIM_THREADS=${NAVSIM_THREADS:-16}
# TOKENS_FILE restricts scoring to a subset via a scene_filter.tokens=[...] hydra override on the command line; only
# "nav" (the 2000-token diagnostic) is a subset. A full-benchmark run dir (navfull, navhard) already predicts every
# token of its split, so it must NOT set TOKENS_FILE: navsim_zs_score.sh's per-token override for 12146+ tokens blows
# past the shell's ARG_MAX ("Argument list too long") -- and it would be a no-op restriction anyway.
# SUBSET=1 marks another subset run dir (op_lb's lb_navtrain, 3 000 navtrain tokens: within ARG_MAX); CACHE_NAME
# (navsim_zs_score.sh) then names its own metric cache.
if [[ $data == nav || -n ${SUBSET:-} ]]; then export TOKENS_FILE=$d/tokens.txt; else unset TOKENS_FILE; fi
# name includes the run dir except for the original "nav" (2000-token) diagnostic, kept as-is for backward
# compatibility: pose files of the same stem (e.g. warp-cinque__base) exist in more than one run dir (nav's
# 2000-token subset vs navfull's/navhard's full set) and must not share one eval directory -- they score different
# token sets under the same stem.
[[ $data == nav ]] && pfx=opi_ || pfx=opi_${data}_
for f in "$d"/preds/*.npz; do
  name=${pfx}$(basename "$f" .npz)
  ls "$DATA_DIR"/runs/navsim/eval/${ver}_${split}_"$name"/*/*.csv >/dev/null 2>&1 && continue
  echo "== $name $(date +%H:%M:%S)"
  if [[ -n $cpus ]]; then taskset -c "$cpus" "$repo/experiments/zeroshot_openloop/archive/navsim_zs_score.sh" score "$ver" "$split" "$name" "$f" > "$d/score_$name.log" 2>&1
  else "$repo/experiments/zeroshot_openloop/archive/navsim_zs_score.sh" score "$ver" "$split" "$name" "$f" > "$d/score_$name.log" 2>&1; fi
  tail -n 3 "$d/score_$name.log"
done
echo "all scored $(date +%H:%M:%S)"
