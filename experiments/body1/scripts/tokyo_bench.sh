#!/usr/bin/env bash
# One timed training run of the P2H10S recipe on one card of the Tokyo box, with what the trainer does not log: the card's peak memory as
# nvidia-smi sees it, the process' peak RSS and the host's page cache, sampled every 2 s.
#   tokyo_bench.sh <name> <card> [bd4_train.py train flags...]      runs bd4_train.py train --tag TB-<name> with the recipe's flags + yours
#   tmux new-window -d -t jev -n tb "bash experiments/body1/scripts/tokyo_bench.sh bank 0 --label-bank --steps 300"
# Env: DATA_DIR (/data), PY ($DATA_DIR/envs/op-train/bin/python), OUT ($DATA_DIR/runs/tokyo_dual3090/train), DATA (the --data list; default
#   navtrain_full.s2of12 twelve times: one pulled shard repeated to the row count of the full run, 281k Store rows, so the label and
#   teacher tensors on the card have their full-run size), RECIPE (the P2H10S flags).
# Writes $OUT/<name>/{STATUS, DONE | ERROR, log.txt, usage.tsv (t, card MiB, card util %, RSS MiB, page cache MiB), bench.json}.
set -uo pipefail
cd "$(dirname "$0")/../../.."
N=$1 G=$2; shift 2
export DATA_DIR=${DATA_DIR:-/data}; PY=${PY:-$DATA_DIR/envs/op-train/bin/python}; O=${OUT:-$DATA_DIR/runs/tokyo_dual3090/train}/$N
DATA=${DATA:-$(printf 'navtrain_full.s2of12 %.0s' {1..12})}
RECIPE=${RECIPE:---ho ot1:4,yr1:4,bd4:5 --ho-w 3 --ho-excl navsim/body1-val-logs --agent-lam 10 --shape}
mkdir -p "$O"; rm -f "$O"/{DONE,ERROR}; st() { echo "$(date '+%F %T') $*" | tee "$O/STATUS"; }
st "running on card $G"
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=$G "$PY" experiments/body1/scripts/bd4_train.py train --tag "TB-$N" --data $DATA $RECIPE "$@" > "$O/log.txt" 2>&1 & pid=$!
: > "$O/usage.tsv"
while kill -0 $pid 2>/dev/null; do
  echo "$(date +%s) $(nvidia-smi -i "$G" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ,) $(awk '/VmRSS/ {r = int($2 / 1024)} END {print r + 0}' /proc/$pid/status 2>/dev/null || echo 0) $(awk '/^Cached:/ {print int($2 / 1024)}' /proc/meminfo)" >> "$O/usage.tsv"
  sleep 2
done
wait $pid; rc=$?
its=$(grep -o '[0-9.]* it/s' "$O/log.txt" | tail -1 | cut -d' ' -f1); gb=$(grep -o '[0-9.]* GB$' "$O/log.txt" | tail -1 | cut -d' ' -f1)
awk -v n="$N" -v g="$G" -v rc=$rc -v its="${its:-0}" -v gb="${gb:-0}" -v args="$*" '{ if ($2 > c) c = $2; u += $3; if ($4 > r) r = $4; if ($5 > p) p = $5 }
  END { printf "{\"name\": \"%s\", \"card\": %s, \"rc\": %s, \"it_s\": %s, \"torch_peak_reserved_gb\": %s, \"card_peak_mib\": %d, \"card_util_mean\": %.0f, \"rss_peak_mib\": %d, \"page_cache_peak_mib\": %d, \"flags\": \"%s\"}\n", n, g, rc, its, gb, c, u / (NR ? NR : 1), r, p, args }' "$O/usage.tsv" > "$O/bench.json"
if (( rc )); then st "ERROR rc $rc: $(grep -m1 -E 'OutOfMemory|Error' "$O/log.txt" | cut -c1-200)"; touch "$O/ERROR"; exit 1; fi
date > "$O/DONE"; st "done: $(cat "$O/bench.json")"
