#!/usr/bin/env bash
# Lateral v2 campaign supervisor (GPU box, run inside tmux via scripts/tmux_run.sh).
# Usage: run_campaign.sh <run-root> <workers: 4|8> [perturbation ids, default p00..p10]
# Workers = route groups (4) or route groups x perturbation halves (8); each worker is one
# b2d_controller_validate.py invocation with its own CARLA server, writing <run-root>/<worker>/.
set -euo pipefail
root=$1; workers=$2; ids=${3:-p00,p01,p02,p03,p04,p05,p06,p07,p08,p09,p10}
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
in=$here/inputs
mkdir -p "$root"
[ -e "$root/events.jsonl" ] && { echo "refusing to reuse $root" >&2; exit 1; }
log() { local line; line=$(printf '{"t": %s, "kind": "%s"%s}' "$(date +%s.%N)" "$1" "${2:-}"); echo "$line" | tee -a "$root/events.jsonl"; echo "$(date +%T) $1 ${2:-}" >> "$root/log.txt"; }
groups=(g1 g2 g3 g4)
IFS=, read -r -a all <<< "$ids"
half=$(( (${#all[@]} + 1) / 2 ))
first=$(IFS=,; echo "${all[*]:0:half}"); second=$(IFS=,; echo "${all[*]:half}")
jobs=()
for g in "${groups[@]}"; do
  if [ "$workers" = 8 ]; then jobs+=("$g:a:$first" "$g:b:$second"); else jobs+=("$g:all:$ids"); fi
done
log start ", \"workers\": $workers, \"ids\": \"$ids\", \"commit\": \"$(git -C "$repo" rev-parse HEAD)\""
pids=(); k=0
for job in "${jobs[@]}"; do
  IFS=: read -r g part pset <<< "$job"
  name=$g-$part; index=$((140 + 3 * k))
  OPENBLAS_CORETYPE=Barcelona PYTHONDONTWRITEBYTECODE=1 "$DATA_DIR/envs/carla/bin/python" "$repo/scripts/b2d_controller_validate.py" \
    --routes "$in/routes-$g.xml" --variants "$in/variants.json" --route-cruises "$in/route-cruises.json" \
    --perturbations "$in/perturbations.json" --perturbation-ids "$pset" --allow-truth-pose-diagnostic \
    --out "$root/$name" --server-index "$index" > "$root/$name.stdout" 2>&1 &
  pids+=($!); log worker_start ", \"worker\": \"$name\", \"pid\": $!, \"server_index\": $index, \"ids\": \"$pset\""
  k=$((k + 1)); sleep 20
done
status=0; k=0
for pid in "${pids[@]}"; do
  if wait "$pid"; then rc=0; else rc=$?; status=1; fi
  log worker_end ", \"worker\": \"${jobs[$k]%%:*}-$(cut -d: -f2 <<< "${jobs[$k]}")\", \"rc\": $rc"; k=$((k + 1))
done
log end ", \"status\": $status"
exit $status
