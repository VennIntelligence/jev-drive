#!/usr/bin/env bash
# Official NAVSIM scoring of the G1 gated arms (todos/2026-09-26-real-data-transfer.md, G1): runs every line of a job
# file written by `python -m jevdrive.real_g1 nav-write` ("<v1|v2> <split> <name> <predictions.npz>") through
# scripts/navsim_zs_score.sh, PAR jobs at a time with NAVSIM_THREADS ray workers each. Logs next to the job file.
#   scripts/real_g1_score.sh <jobs.txt> [PAR=6]
set -euo pipefail
jobs=$1 par=${2:-6}
repo=$(cd "$(dirname "$0")/.." && pwd)
dir=$(dirname "$jobs")
export NAVSIM_THREADS=${NAVSIM_THREADS:-16}
grep -v '^\s*$' "$jobs" | xargs -P "$par" -L 1 bash -c \
  'log='"$dir"'/score_$0_$1_$2.log; "'"$repo"'/scripts/navsim_zs_score.sh" score "$0" "$1" "$2" "$3" > "$log" 2>&1 \
   && echo "done $0 $2" || echo "FAILED $0 $2 (see $log)"'
touch "$dir/scored_$(basename "$jobs" .txt).done"
