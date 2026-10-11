#!/usr/bin/env bash
# Several pai_eval.sh runs through one queue on the cards of the Tokyo box: every card walks the runs in order and takes the next unclaimed
# chunk of the current run, so a card that has nothing left in one run starts the next run at once. Run one after the other, pai_eval.sh alone
# leaves the faster card idle until the slower one has finished its last chunk (a chunk is 25 to 40 min).
#   pai_queue.sh <root> '<name>|<checkpoint tag>|<scene list or glob>[ <more lists>]'...
#   tmux new-window -d -t jev -n pq "bash experiments/alpasim/scripts/pai_queue.sh /data/runs/alpasim/pq \
#     'a|P2H10-F-s1|/data/runs/alpasim/pai_full/s_b2.tsv /data/runs/alpasim/pai_full/s_n1.tsv' 'b|P2H10S-F-s2|/data/runs/alpasim/pai_full/s_*.tsv'"
# Env: CARDS ("0 1"), CHUNK (36), EVAL (pai_eval.sh); CONC, SERVE, IMG, ... reach pai_eval.sh / pai_run.sh unchanged. Globs are expanded here, in
#   the C locale and in the order given, so the chunk cut is reproducible.
# Layout: <root>/<name>/ is the output root of pai_eval.sh (chunks/, claims/, runs/<lowercase tag>_<chunk>/, log.txt); this script adds
#   <root>/{STATUS, DONE | ERROR, log.txt, gpu.csv (card utilisation, power, clock and memory once a minute)}. A rerun resumes: finished chunks are
#   skipped, claims are reset. A chunk that failed twice (pai_eval.sh has already tried it twice) gets one more pass at the end, on whichever
#   card is free; if it fails again, ERROR names it.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
R=$(mkdir -p "$1" && cd "$1" && pwd); shift
EVAL=${EVAL:-$here/pai_eval.sh}; CARDS=${CARDS:-0 1}; CHUNK=${CHUNK:-36}
rm -f "$R"/{DONE,ERROR,.stop}; exec > >(tee -a "$R/log.txt") 2>&1
st() { echo "$(date '+%F %T') $*" | tee "$R/STATUS"; }
(( $# )) || { st "ERROR no run given"; exit 1; }
names=() tags=() lists=()
for j in "$@"; do
  IFS='|' read -r n t l <<< "$j"
  [[ -n $n && -n $t && -n $l ]] || { st "ERROR bad run spec '$j'"; exit 1; }
  l=$( export LC_ALL=C; for p in $l; do compgen -G "$p" || { echo "no scene list matches $p" >&2; exit 1; }; done ) || { st "ERROR scene lists of $n"; exit 1; }
  names+=("$n"); tags+=("$t"); lists+=("$(echo $l)")
done
all=("${!names[@]}"); total=0
for i in "${all[@]}"; do   # chunk lists are written once (pai_eval.sh with no card); the checkpoint is checked here, before any stack starts
  O=$R/${names[i]}; mkdir -p "$O"; rm -rf "$O/claims"; mkdir "$O/claims"; mv -f "$O/ERROR" "$O/ERROR.$(date +%s)" 2>/dev/null
  JOIN=1 CARDS=" " CHUNK=$CHUNK bash "$EVAL" "$O" "${tags[i]}" ${lists[i]} > /dev/null || { st "ERROR init of run ${names[i]}, see $O/log.txt"; exit 1; }
  total=$(( total + $(cat "$O"/chunks/*.txt | wc -l) ))
done
unfinished() {  # run index: its chunks without a DONE
  local i=$1 L f c; L=${tags[i]}; L=${L,,}
  for f in "$R/${names[i]}"/chunks/*.txt; do c=$(basename "$f" .txt); [[ -f $R/${names[i]}/runs/${L}_$c/DONE ]] || echo "${names[i]}/$c"; done
}
walk() {  # card, run indices: pai_eval.sh JOIN=1 on one card takes unclaimed chunks of a run and returns when none is left
  local g=$1 i; shift
  for i in "$@"; do
    echo "$(date '+%F %T') card $g: run ${names[i]}"
    JOIN=1 CARDS=$g CHUNK=$CHUNK bash "$EVAL" "$R/${names[i]}" "${tags[i]}" ${lists[i]}
  done
  echo "$(date '+%F %T') card $g: nothing left"
}
pass() {  # run indices: both cards walk them (staggered: two stacks starting at once race for the docker network)
  local pids=() g
  for g in $CARDS; do walk "$g" "$@" & pids+=($!); sleep "${STAGGER:-25}"; done
  wait "${pids[@]}"
}
( while [[ ! -f $R/.stop ]]; do
    nvidia-smi --query-gpu=index,utilization.gpu,power.draw,clocks.gr,memory.used --format=csv,noheader,nounits 2>/dev/null | sed "s/^/$(date '+%F %T'), /" >> "$R/gpu.csv"
    sleep 60; done ) & sampler=$!
st "queue: ${#names[@]} runs (${names[*]}), $total scenes, cards $CARDS, chunk $CHUNK"
t0=$SECONDS
pass "${all[@]}"
bad=(); for i in "${all[@]}"; do bad+=($(unfinished "$i")); done
if (( ${#bad[@]} )); then
  st "repair pass for ${bad[*]}"
  redo=(); for i in "${all[@]}"; do
    [[ -n $(unfinished "$i") ]] || continue
    redo+=("$i"); O=$R/${names[i]}; rm -rf "$O/claims"; mkdir "$O/claims"; mv -f "$O/ERROR" "$O/ERROR.pass1" 2>/dev/null
  done
  pass "${redo[@]}"
  bad=(); for i in "${redo[@]}"; do bad+=($(unfinished "$i")); done
fi
touch "$R/.stop"; kill "$sampler" 2>/dev/null
w=$(( SECONDS - t0 ))
if (( ${#bad[@]} )); then echo "unfinished chunks: ${bad[*]}" > "$R/ERROR"; st "ERROR unfinished chunks ${bad[*]} after $w s"; exit 1; fi
date > "$R/DONE"; st "done: $total scenes in $w s"
