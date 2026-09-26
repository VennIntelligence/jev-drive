#!/usr/bin/env bash
# Night queue 3, lane C, Q1: the four NAVSIM-trained examinees on the P6 v0 exam frames, run as they ship
# (todos/2026-09-26-night-queue-3.md, [C-nav] entries). SparseDriveV2 / ZTRS through the T1 adapter
# (jevdrive.top10_exam plan --set p6 + scripts/top10_exam_infer.py), DrivoR / WA-JEPA through the T2 request path
# (jevdrive.top10_t2 req-p6 + scripts/top10_t2/{drivor,wajepa}_run.py). Outputs processed/top10_exam/p6/<model>.npz
# (frame_name, raw, grid (n, 20, 2) rear-axle ego frame, 0.25 s steps), readable by top10_exam.load_preds.
#
# Usage (on the box, in tmux): scripts/tmux_run.sh nq3-c-nav scripts/nq3_c_nav.sh
#   PRI=0,1   frame-list priorities (the [C-nav] entry fixes 0,1)
# Idempotent: a model whose output covers the planned frames is skipped; WA-JEPA and DrivoR resume from their chunks.
# One invocation at a time (flock); a second one waits and then sees DONE. Pins itself to cores 150-157 and GPU 6.
# GPU 6 is shared: every GPU job waits for enough free VRAM first and is retried after a failure (OOM from a
# neighbour); our own total stays <= ~30 GB (T1 lane <= 20, DrivoR 7, 2 + 6 WA-JEPA processes x 3 GB).
set -euo pipefail
: "${DATA_DIR:?}"
repo=$(cd "$(dirname "$0")/.." && pwd)
PRI=${PRI:-0,1}
R=$DATA_DIR/runs/nq3/c/q1_nav
OUT=$DATA_DIR/processed/top10_exam/p6
T2=$DATA_DIR/runs/top10_t2
SH=$T2/preds/p6_wajepa_shards
NCHUNK=48
mkdir -p "$R/logs" "$OUT" "$SH"
exec 9> "$R/lock"
flock 9
[[ -f $R/DONE ]] && { echo "already done: $R/DONE"; exit 0; }
rm -f "$R/FAILED"
taskset -cp 150-157 $$ > /dev/null
export CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$DATA_DIR/envs/jevdrive/bin/python
t0=$SECONDS
log() { echo "$(date '+%F %T') $*" | tee -a "$R/log.txt"; }
fail() { log "FAILED: $*"; echo "{\"failed\": \"$*\", \"at\": \"$(date '+%F %T')\"}" > "$R/FAILED"; exit 1; }
trap 'fail "line $LINENO"' ERR

# ---------------------------------------------------------------- plan and requests (CPU, seconds)
cd "$repo"
$PY -m jevdrive.top10_exam plan --set p6 --priorities "$PRI" >> "$R/logs/plan.log" 2>&1
$PY -m jevdrive.top10_t2 req-p6 --priorities "$PRI" >> "$R/logs/plan.log" 2>&1
sha=$(cat "$T2/requests/p6_drivor.npz" "$T2/requests/p6_wajepa.npz" | sha1sum | cut -c1-16)
if [[ $(cat "$R/requests.sha" 2>/dev/null) != "$sha" ]]; then     # new frame set: drop partial outputs of an old one
  rm -rf "$T2/preds/p6_drivor.npz.part" "$SH"; mkdir -p "$SH"; echo "$sha" > "$R/requests.sha"
fi
rm -rf "$SH"/claim.*                                             # we hold the lock: no live claims
n_plan=$($PY -c "import json;print(len(json.load(open('$OUT/plan.json'))['frames']['frame_name']))")
log "start: priorities $PRI, $n_plan frames, requests $sha"

complete() {  # <npz>: its frame_name set equals the plan's
  [[ -f $1 ]] && $PY - "$1" "$OUT/plan.json" <<'EOF'
import json, sys, numpy as np
z = np.load(sys.argv[1], allow_pickle=True)
sys.exit(0 if set(z["frame_name"].astype(str)) == set(json.load(open(sys.argv[2]))["frames"]["frame_name"]) else 1)
EOF
}

wait_vram() {  # <GB>: sleep until GPU 6 has this much free
  until (( $(nvidia-smi -i 6 --query-gpu=memory.free --format=csv,noheader,nounits) >= $1 * 1024 )); do sleep 30; done
}

retry() {  # <name> <GB> <cmd...>: wait for VRAM, run, up to 6 attempts (T2 runners resume from their chunks)
  local name=$1 gb=$2 k; shift 2
  for k in 1 2 3 4 5 6; do
    wait_vram "$gb"
    log "$name: attempt $k"
    if "$@" >> "$R/logs/$name.log" 2>&1; then log "$name: ok"; return 0; fi
    log "$name: attempt $k failed (see logs/$name.log)"; sleep 60
  done
  return 1
}

t1() {  # <model> <batch> <GB>
  complete "$OUT/$1.npz" && { log "$1: complete, skipped"; return 0; }
  retry "$1" "$3" "$repo/scripts/top10_exam.sh" "$1" --set p6 --workers 5 --batch "$2"
}

drivor() {
  complete "$OUT/drivor.npz" && { log "drivor: complete, skipped"; return 0; }
  retry drivor 8 bash -c "cd $DATA_DIR/third_party/drivor && $DATA_DIR/envs/drivor/bin/python \
    $repo/scripts/top10_t2/drivor_run.py $T2/requests/p6_drivor.npz --out $T2/preds/p6_drivor.npz --workers 3 --bs 16"
}

wj_worker() {  # claims WA-JEPA shards (mkdir is atomic) until none is left; a failed shard is released and retried
  local w=$1 i k
  for k in 1 2 3; do
    for i in $(seq 0 $((NCHUNK - 1))); do
      [[ -f $SH/$i.npz ]] && continue
      mkdir "$SH/claim.$i" 2> /dev/null || continue
      wait_vram 4
      if (cd "$DATA_DIR/third_party/wajepa" && "$DATA_DIR/envs/wajepa/bin/python" "$repo/scripts/top10_t2/wajepa_run.py" \
            "$T2/requests/p6_wajepa.npz" --out "$SH/.$i.tmp.npz" --shard "$i" "$NCHUNK" --workers 1 --cache "$R/wajepa_cache" \
            >> "$R/logs/wajepa.w$w.log" 2>&1); then
        mv "$SH/.$i.tmp.npz" "$SH/$i.npz"
      else
        log "wajepa shard $i failed on worker $w"; sleep 60
      fi
      rmdir "$SH/claim.$i"
    done
  done
}

wajepa_done() { (( $(ls "$SH"/[0-9]*.npz 2> /dev/null | wc -l) == NCHUNK )); }

# ---------------------------------------------------------------- WA-JEPA image cache (CPU, every unique image once)
if ! complete "$OUT/wajepa.npz"; then
  (cd "$DATA_DIR/third_party/wajepa" && "$DATA_DIR/envs/wajepa/bin/python" "$repo/scripts/top10_t2/wajepa_run.py" \
     "$T2/requests/p6_wajepa.npz" --cache "$R/wajepa_cache" --cache-only --workers 8 >> "$R/logs/wajepa_cache.log" 2>&1)
  log "wajepa cache ready"
fi

# ---------------------------------------------------------------- GPU lanes
set +e; trap - ERR
lane_t() { t1 sparsedrivev2 16 16 && t1 ztrs 32 21; local rc=$?; complete "$OUT/wajepa.npz" || {
  for w in 2 3 4 5 6 7; do wj_worker "$w" & done; wait; }; return $rc; }
lane_w() { drivor; local rc=$?; complete "$OUT/wajepa.npz" || { for w in 0 1; do wj_worker "$w" & done; wait; }; return $rc; }
lane_t & pt=$!
lane_w & pw=$!
wait $pt; rt=$?
wait $pw; rw=$?
set -e; trap 'fail "line $LINENO"' ERR
(( rt == 0 )) || fail "T1 lane (see logs/sparsedrivev2.log, logs/ztrs.log)"
(( rw == 0 )) || fail "DrivoR (see logs/drivor.log)"

# ---------------------------------------------------------------- merge, export, sanity, DONE
if ! complete "$OUT/wajepa.npz"; then
  wajepa_done || fail "WA-JEPA: $(ls "$SH"/[0-9]*.npz 2> /dev/null | wc -l) / $NCHUNK shards"
  (cd "$DATA_DIR/third_party/wajepa" && "$DATA_DIR/envs/wajepa/bin/python" "$repo/scripts/top10_t2/wajepa_run.py" \
     --merge $(for i in $(seq 0 $((NCHUNK - 1))); do echo "$SH/$i.npz"; done) --out "$T2/preds/p6_wajepa.npz")
fi
$PY -m jevdrive.top10_t2 export-p6 >> "$R/logs/plan.log" 2>&1
for m in sparsedrivev2 ztrs drivor wajepa; do complete "$OUT/$m.npz" || fail "$m output incomplete"; done
$PY "$repo/scripts/nq3_c_nav_check.py" sanity > "$R/sanity.json" 2>> "$R/logs/plan.log"
rm -rf "$R/wajepa_cache"
$PY - "$R" "$OUT" "$((SECONDS - t0))" "$PRI" <<'EOF'
import json, sys, time, numpy as np
r, out, wall, pri = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
ms = ("sparsedrivev2", "ztrs", "drivor", "wajepa")
d = {"finished": time.strftime("%F %T"), "wall_s": wall, "priorities": pri,
     "outputs": {m: f"{out}/{m}.npz" for m in ms},
     "frames": {m: int(len(np.load(f"{out}/{m}.npz", allow_pickle=True)["frame_name"])) for m in ms},
     "format": "frame_name (n,), raw (model's own poses), grid (n, 20, 2) at 0.25..5 s, rear-axle ego frame, x fwd, y left, m",
     "sanity": json.load(open(f"{r}/sanity.json"))}
json.dump(d, open(f"{r}/DONE", "w"), indent=1)
print(json.dumps(d, indent=1))
EOF
log "done in $((SECONDS - t0)) s"
