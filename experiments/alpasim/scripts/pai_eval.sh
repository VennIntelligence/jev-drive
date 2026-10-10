#!/usr/bin/env bash
# PAI curated-validation read of one checkpoint on every card of the Tokyo box. The renderer is the bottleneck (81 % of the rollout time,
# docs/tokyo-box.md "Two cards"), it is bound to one card and one stack fills that card's 24 GB, so the scenes are cut into chunks and every
# card runs its own pai_run.sh stack (own compose project, ports, driver container) on the next unclaimed chunk until none is left.
#   pai_eval.sh <out root> <checkpoint tag> <scene list>...      lists: pai_scenes.py tsv (stage <= 1 rows) or plain scene ids
#   tmux new-window -d -t jev -n pai "bash experiments/alpasim/scripts/pai_eval.sh /data/runs/alpasim/pai_full2 P2H10S-F-s1 /data/runs/alpasim/pai_full/s_*.tsv"
# Env: CARDS ("0 1"), CONC (4 rollouts per card), NRE_CACHE (pai_run.sh), CHUNK (scenes per stack start, 48: a start costs ~2 min, a failed
#   chunk is rerun once and loses at most its own scenes), SERVE (driver switches, default the served configuration
#   "-e JEV_VCONT=1.0 -e JEV_LEAD=1"), IMG / DRV_PY / HARMONIZER as pai_run.sh. The checkpoint is mounted into the driver image.
#   JOIN=1: these cards join a run that is already going on <out root> (same tag and lists): they take unclaimed chunks and leave STATUS,
#   DONE / ERROR and eval.json to the first call. A card freed by another job is added this way.
# State: <out>/{STATUS, DONE | ERROR, log.txt, eval.json (scenes, wall, scenes per hour)}, chunk lists in <out>/chunks/, one run per chunk in
# <out>/runs/<lowercase tag>_<chunk> (finished chunks are skipped: rerunning resumes). Read with pai_eval_report.py.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
O=$(mkdir -p "$1" && cd "$1" && pwd) T=$2; shift 2
D=${DATA_DIR:-/data}; C=$D/runs/op_parity/runs/$T; L=${T,,}
CARDS=${CARDS:-0 1}; CONC=${CONC:-4}; CHUNK=${CHUNK:-48}; SERVE=${SERVE--e JEV_VCONT=1.0 -e JEV_LEAD=1}
J=${JOIN:-0}; mkdir -p "$O"/{chunks,runs}; (( J )) || { rm -rf "$O"/{DONE,ERROR,claims}; mkdir "$O/claims"; }; exec > >(tee -a "$O/log.txt") 2>&1
st() { echo "$(date '+%F %T') $T: $*" | if (( J )); then cat; else tee "$O/STATUS"; fi; }
[[ -f $C/ckpt-final.pt ]] || { st "ERROR no checkpoint $C"; touch "$O/ERROR"; exit 1; }
if ! ls "$O"/chunks/*.txt >/dev/null 2>&1; then           # written once: a resumed run keeps its chunks
  for f in "$@"; do if head -1 "$f" | grep -q '^scene_id'; then awk -F'\t' 'NR > 1 && $8 <= 1 {print $1}' "$f"; else grep . "$f"; fi; done \
    | awk '!seen[$0]++' | split -l "$CHUNK" -d -a 3 --additional-suffix=.txt - "$O/chunks/c"
fi
n=$(cat "$O"/chunks/*.txt | wc -l); t0=$(date +%s)
st "$n scenes in $(ls "$O"/chunks | wc -l) chunks, cards $CARDS, $CONC concurrent per card"
worker() {  # card: takes chunks until none is left; a card that reads the 2026-10-10 fault (idle but > 300 W) leaves the rest to the others
  local g=$1 f c r k
  for f in "$O"/chunks/*.txt; do
    c=$(basename "$f" .txt); r=$O/runs/${L}_$c
    mkdir "$O/claims/$c" 2>/dev/null || continue
    [[ -f $r/DONE ]] && continue
    read -r pw ut < <(nvidia-smi -i "$g" --query-gpu=power.draw,utilization.gpu --format=csv,noheader,nounits | tr -d ,)
    if (( ${ut%.*} < 10 && ${pw%.*} > 300 )); then echo "card $g idle at $pw W" >> "$O/ERROR"; rmdir "$O/claims/$c"; return 1; fi
    for k in 1 2; do
      echo "$(date '+%F %T') card $g: $c ($(wc -l < "$f") scenes), attempt $k"
      GPU=$g CONC=$CONC PAI_BASEPORT=$((6400 + 100 * g)) TAG=$T DRV_ENV="$SERVE -v $C:/app/data/runs/op_parity/runs/$T:ro" \
        bash "$here/pai_run.sh" "$r" "$f" > "$O/runs/${L}_$c.out" 2>&1 && break
      (( k == 2 )) && echo "chunk $c failed twice on card $g: $(cat "$r/ERROR" 2>/dev/null)" >> "$O/ERROR"
    done
  done
}
pids=(); for g in $CARDS; do worker "$g" & pids+=($!); sleep 20; done      # staggered: two stacks starting at once race for the wizard's docker network
wait "${pids[@]}"                                                         # the workers only: a bare wait also waits for the log tee above, which never ends
(( J )) && { st "joined cards $CARDS: no chunk left"; exit 0; }
busy() {  # a claimed chunk of a joined card is still running (its driver container is up)
  local c; for c in "$O"/claims/*; do c=$(basename "$c"); [[ -f $O/runs/${L}_$c/DONE ]] && continue
    docker ps -q --filter "name=pai-$(echo "${L}_$c" | tr -c 'a-zA-Z0-9\n' '-')-drv" | grep -q . && return 0; done; return 1; }
while busy || { sleep 60; busy; }; do sleep 30; done
left=$(for f in "$O"/chunks/*.txt; do [[ -f $O/runs/${L}_$(basename "$f" .txt)/DONE ]] || echo "$f"; done | wc -l)
(( left > 0 )) && ! [[ -f $O/ERROR ]] && echo "$left chunks unfinished" >> "$O/ERROR"
w=$(( $(date +%s) - t0 ))
echo "{\"tag\": \"$T\", \"scenes\": $n, \"cards\": \"$CARDS\", \"conc\": $CONC, \"chunk\": $CHUNK, \"wall_s\": $w, \"scenes_per_h\": $(( n * 3600 / (w + 1) )), \"chunks_left\": $left}" > "$O/eval.json"
if [[ -f $O/ERROR ]]; then st "ERROR $(tail -1 "$O/ERROR")"; exit 1; fi
date > "$O/DONE"; st "done: $n scenes in $w s"
