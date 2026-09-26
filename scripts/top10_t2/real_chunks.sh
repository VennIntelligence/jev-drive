#!/usr/bin/env bash
# Resumable WA-JEPA inference on one T2 request set: the requests are cut into NCHUNK shards (wajepa_run.py --shard i
# NCHUNK); LANES processes work through the missing shards; each finished shard is its own file, so a stopped run
# resumes by rerunning the same command. No new shard starts after DEADLINE (HH:MM box clock, optional). When all
# shards exist they are merged into $DATA_DIR/runs/top10_t2/preds/<set>_wajepa.npz.
# Usage: GPU=3 LANES=2 NCHUNK=20 WORKERS=2 [DEADLINE=11:31] scripts/top10_t2/real_chunks.sh <set>
set -uo pipefail
: "${DATA_DIR:?}"
set_=$1
GPU=${GPU:-3}; LANES=${LANES:-2}; NCHUNK=${NCHUNK:-20}; WORKERS=${WORKERS:-2}; DEADLINE=${DEADLINE:-}
repo=$(cd "$(dirname "$0")/../.." && pwd)
R=$DATA_DIR/runs/top10_t2; req=$R/requests/$set_.npz; out=$R/preds/chunks_$set_; mkdir -p "$out" "$R/logs"
lane() {
  for i in $(seq "$1" "$LANES" $((NCHUNK - 1))); do
    [[ -f $out/$i.npz ]] && continue
    if [[ -n $DEADLINE && $(date +%H:%M) > $DEADLINE ]]; then echo "lane $1: deadline, stop before shard $i"; return; fi
    (cd "$DATA_DIR/third_party/wajepa" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2 "$DATA_DIR/envs/wajepa/bin/python" \
      "$repo/scripts/top10_t2/wajepa_run.py" "$req" --out "$out/.$i.tmp.npz" --shard "$i" "$NCHUNK" --workers "$WORKERS" \
      > "$R/logs/${set_}_wajepa.chunk$i.log" 2>&1) && mv "$out/.$i.tmp.npz" "$out/$i.npz" && echo "$(date +%T) shard $i done"
  done
}
for l in $(seq 0 $((LANES - 1))); do lane "$l" & done
wait
n=$(ls "$out"/[0-9]*.npz 2>/dev/null | wc -l)
echo "$n / $NCHUNK shards"
if (( n == NCHUNK )); then
  (cd "$DATA_DIR/third_party/wajepa" && "$DATA_DIR/envs/wajepa/bin/python" "$repo/scripts/top10_t2/wajepa_run.py" \
    --merge $(for i in $(seq 0 $((NCHUNK - 1))); do echo "$out/$i.npz"; done) --out "$R/preds/${set_}_wajepa.npz") \
    && echo "merged -> $R/preds/${set_}_wajepa.npz"
fi
