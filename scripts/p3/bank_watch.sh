#!/usr/bin/env bash
# Donor bank expansion (user 2026-09-28, (b)): bank every reconstructed scene - the pedestrian pool (ckpt/nq4_p3/p3/<k>)
# and the vehicle-deletion segments (ckpt/nq4_p3_veh/p3/<j>, donor scene id 1000 + j) - two jobs at a time on shared cards
# with >= 20 GB free (a bank job holds ~12 GB with CPU preload; never GPU 1: Cosmos demand; not GPU 6: the dose renders), re-scanning every 20 min for new
# reconstructions until the P3 lane writes expand/DONE. Markers in runs/nq4/p3/xinsert/bank/: PASS1_DONE after the first
# full pass, WATCH_DONE at the end, <name>.fail for a job that failed (not retried; the watcher goes on).
#   scripts/tmux_run.sh p3-bank bash scripts/p3/bank_watch.sh
set -uo pipefail
cd ~/data/jev-drive
X=$DATA_DIR/runs/nq4/p3/xinsert; B=$X/bank; L=$X/bank_watch.log
PY=$DATA_DIR/envs/drivestudio/bin/python
CARDS=(2 3 4 5); CORES=("176,177" "178,179")
log() { echo "$(date '+%F %T') $*" | tee -a "$L"; }

cands() {   # "id name" of reconstructions with a final checkpoint and no bank file / fail marker
  for d in "$DATA_DIR"/ckpt/nq4_p3/p3/[0-9][0-9][0-9]; do
    n=p3_$(basename "$d"); [ -f "$d/checkpoint_final.pth" ] && [ ! -f "$B/$n.json" ] && [ ! -f "$B/$n.fail" ] && echo "$((10#$(basename "$d"))) $n"
  done
  for d in "$DATA_DIR"/ckpt/nq4_p3_veh/p3/[0-9][0-9][0-9]; do
    n=v3_$(basename "$d"); [ -f "$d/checkpoint_final.pth" ] && [ ! -f "$B/$n.json" ] && [ ! -f "$B/$n.fail" ] && echo "$((1000 + 10#$(basename "$d"))) $n"
  done
}

pick_gpu() {
  while true; do
    for g in "${CARDS[@]}"; do
      free=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
      [ -n "$free" ] && [ "$free" -ge 20000 ] && { echo "$g"; return; }
    done
    sleep 60
  done
}

job() {   # id name cores
  local g
  # one start at a time, 120 s apart, so the second job reads the free VRAM after the first one has allocated
  exec 9>"$B/.start.lock"; flock 9; g=$(pick_gpu)
  log "bank $2 (scene id $1) on GPU $g"
  ( sleep 120; flock -u 9 ) &
  exec 9>&-
  if CUDA_VISIBLE_DEVICES=$g P3_PRELOAD_DEVICE=cpu TORCH_EXTENSIONS_DIR=$DATA_DIR/cache/torch_ext_p3x OMP_NUM_THREADS=2 \
      taskset -c "$3" timeout 3600 "$PY" scripts/p3/xinsert.py bank --scene "$1" > "$X/bank_$2.out" 2>&1; then
    log "bank $2 done: $(tail -1 "$X/bank_$2.out")"
  else
    log "bank $2 FAILED, see $X/bank_$2.out"; touch "$B/$2.fail"
  fi
}

summary() {
  (cd scripts/p3 && "$PY" -c "
import json, numpy as np, xinsert as XI
b = XI.load_bank(); dc = np.array([d['d_close'] for d in b])
s = {'donors': len(b), 'files': len(list((XI.X / 'bank').glob('*.json'))), 'd_close_le': {x: int((dc <= x).sum()) for x in (5, 7, 9, 12, 16, 20, 30)}}
(XI.X / 'bank' / 'summary.json').write_text(json.dumps(s)); print(json.dumps(s))") >> "$L" 2>&1
}

log "bank watcher started, pid $$"
pass=0
while true; do
  mapfile -t todo < <(cands)
  if [ ${#todo[@]} -gt 0 ]; then
    log "${#todo[@]} to bank: ${todo[*]}"
    i=0
    for c in "${todo[@]}"; do
      set -- $c
      job "$1" "$2" "${CORES[$((i % 2))]}" &
      i=$((i + 1))
      [ $((i % 2)) -eq 0 ] && wait
    done
    wait
    continue                                  # re-scan at once: reconstructions may have finished meanwhile
  fi
  pass=$((pass + 1)); summary
  [ $pass -eq 1 ] && touch "$B/PASS1_DONE" && log "first pass done"
  if [ -f "$DATA_DIR/runs/nq4/p3/expand/DONE" ]; then touch "$B/WATCH_DONE"; log "lane done, watcher exits"; exit 0; fi
  sleep 1200
done
