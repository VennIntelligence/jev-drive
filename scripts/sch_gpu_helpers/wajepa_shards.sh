#!/usr/bin/env bash
# SCH GPU helper: extra WA-JEPA shard workers for lane C's NAVSIM batch (scripts/nq3_c_nav.sh, wj_worker) on an idle card.
# Same claim protocol (mkdir $SH/claim.$i), same command, same outputs (.$i.tmp.npz -> $i.npz) as wj_worker.
# Walks the shards from the END (the lane's workers walk from the start) and claims a shard only while at least
# MARGIN other shards stay unclaimed and undone: the lane's workers then still start a shard after this claim, so
# the lane's `wajepa_done` check (all 48 shards after its workers return) cannot run while a helper holds a claim,
# as long as a helper shard is faster than a lane shard (checked by the caller; idle card vs shared GPU 6).
# Usage: CUDA_VISIBLE_DEVICES=<card> wajepa_shards.sh <tag> [MARGIN]
set -uo pipefail
: "${DATA_DIR:?}"
tag=$1 MARGIN=${2:-2}
repo=$(cd "$(dirname "$0")/../.." && pwd)
R=$DATA_DIR/runs/nq3/c/q1_nav
T2=$DATA_DIR/runs/top10_t2
SH=$T2/preds/p6_wajepa_shards
NCHUNK=48
O=$DATA_DIR/runs/sched/gpu_helpers/wajepa
mkdir -p "$O"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') [$tag] $*" | tee -a "$O/log.txt"; }
free_shards() { local j n=0; for ((j = 0; j < NCHUNK; j++)); do [[ -f $SH/$j.npz || -d $SH/claim.$j ]] || n=$((n + 1)); done; echo $n; }
[[ -f $R/DONE || -f $R/FAILED ]] && { log "lane batch already DONE/FAILED, nothing to do"; exit 0; }
cur=""   # the claim held right now: released on any exit so the lane's workers can take the shard
trap '[[ -n $cur ]] && rmdir "$SH/claim.$cur" 2> /dev/null && log "released claim $cur on exit"' EXIT
trap 'exit 143' TERM INT
log "start on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, margin $MARGIN"
n_done=0
for ((i = NCHUNK - 1; i >= 0; i--)); do
  [[ -f $SH/$i.npz ]] && continue
  (( $(free_shards) - 1 >= MARGIN )) || { log "stop: fewer than $MARGIN unclaimed shards would remain for the lane"; break; }
  mkdir "$SH/claim.$i" 2> /dev/null || continue
  cur=$i
  [[ -f $SH/$i.npz ]] && { rmdir "$SH/claim.$i"; cur=""; continue; }
  t0=$SECONDS
  if (cd "$DATA_DIR/third_party/wajepa" && "$DATA_DIR/envs/wajepa/bin/python" "$repo/scripts/top10_t2/wajepa_run.py" \
        "$T2/requests/p6_wajepa.npz" --out "$SH/.$i.tmp.npz" --shard "$i" "$NCHUNK" --workers 1 --cache "$R/wajepa_cache" \
        >> "$O/$tag.log" 2>&1); then
    mv "$SH/.$i.tmp.npz" "$SH/$i.npz"
    n_done=$((n_done + 1))
    log "shard $i done in $((SECONDS - t0)) s"
  else
    log "shard $i failed after $((SECONDS - t0)) s (released for the lane)"
  fi
  rmdir "$SH/claim.$i"; cur=""
done
log "exit: $n_done shards done"
