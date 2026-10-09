#!/usr/bin/env bash
# Lane FIX1, PAI track: one review GIF per zero-score rollout of the queue q.sh, made by COL1's clip pipeline (col1_pai_extract.py ->
# col1_pai_replay.py on the zero scenes only -> col1_pai.py -> col1_pai_clip.py; reused as shipped). Runs on the Tokyo box from a code copy:
# /data/runs/alpasim/fix1/code_clips/{scripts,COMMIT}; COL1's code dir supplies the replay driver (its lib is mounted as in col1_pai_chain.sh).
#   tmux new-window -d -t jev -n fix1-clips 'bash /data/runs/alpasim/fix1/code_clips/scripts/fix1_pai_clips.sh'
# Waits for $F/q/DONE, so no card is touched while the queue runs. Per finished run (fx_*, DONE without "skipped"): extract on CPU, replay of the
# zero scenes on a card (two workers, one per card, a run is claimed by mkdir), analysis + clip on CPU. Nothing outside $OUT is written
# (COL1's and pai1's dirs and the queue's runs are read only). State: $OUT/{STATUS, DONE | ERROR, log.txt, index.tsv, gif/, x/<run>/}.
# The replay is open loop through the plain baseline driver on the logged driver-side messages: the plans drawn are the checkpoint's, not
# necessarily the arm's served plan; the bird's-eye ego path, actors, gap and speed are the run's own.
# Test knobs: RUNROOT (default $F/runs), RUNS (names), OUT, NOWAIT=1 (skip the q/DONE wait), PRE=<dir with <run>/{logs.pkl,replay}> (reuse an
# extract and replay instead of making them).
set -uo pipefail
F=/data/runs/alpasim/fix1; C=/data/runs/alpasim/col1; CODE=${CODE:-$F/code_clips}; OUT=${OUT:-$F/clips}; RUNROOT=${RUNROOT:-$F/runs}
IMG=jev-alpasim:p2h10-f-s0-6f3d05d1; BASE=alpasim-base:0.89.0; H=$CODE/scripts/fix1_pai_clips.py
rm -rf "$OUT/DONE" "$OUT/ERROR" "$OUT/claim" "$OUT/fail"; mkdir -p "$OUT/gif" "$OUT/x" "$OUT/claim" "$OUT/fail" "$OUT/idx"
exec > >(tee -a "$OUT/log.txt") 2>&1
st() { echo "$(date '+%F %T') $*" | tee "$OUT/STATUS"; }
card_free() {  # card index: wait until it has held under 2 GiB for a minute
  local ok=0; until (( ok >= 6 )); do (( $(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits) < 2048 )) && ok=$((ok + 1)) || ok=0; sleep 10; done; }
dk() {  # trusted image, no GPU: python script + args
  docker run --rm -v /data:/data -e PYTHONPATH=/data/third_party/alpasim/src/utils -e HF_HUB_OFFLINE=1 -e MPLCONFIGDIR=/tmp/mpl $BASE bash -c "umask 0000; cd /repo && uv run python $*"; }
done_real() { [[ -f $RUNROOT/$1/DONE ]] && ! grep -q skipped "$RUNROOT/$1/DONE"; }

st "code $(cat "$CODE/COMMIT" 2>/dev/null); waiting for $F/q/DONE"
[[ -n ${NOWAIT:-} ]] || until [[ -f $F/q/DONE ]]; do sleep 30; done

if [[ -n ${RUNS:-} ]]; then R=($RUNS); else R=(); for d in "$RUNROOT"/fx_*; do [[ -d $d ]] && done_real "$(basename "$d")" && R+=("$(basename "$d")"); done; fi
st "${#R[@]} finished runs"

one() {  # run name, card -> $OUT/gif/<arm>_<list>_<scene8>_<why>.gif, $OUT/idx/<run>.tsv
  local r=$1 g=$2 n arm l X=$OUT/x/$1 s sc
  n=${r#fx_}; arm=${n%%_*}; l=${n#*_}; mkdir -p "$X"
  python3 "$H" zeros --run "$RUNROOT/$r" > "$X/zeros.tsv" || return 1
  [[ -s $X/zeros.tsv ]] && echo "$(date '+%F %T') $r: $(wc -l < "$X/zeros.tsv") zero rollouts" || { echo "$(date '+%F %T') $r: no zeros"; : > "$OUT/idx/$r.tsv"; return 0; }
  if [[ -n ${PRE:-} ]]; then cp "$PRE/$r/logs.pkl" "$X/"; mkdir -p "$X/replay"; cp "$PRE/$r/replay/"*.npz "$X/replay/"
  else
    if [[ ! -f $X/logs.pkl ]]; then
      dk "$CODE/scripts/col1_pai_extract.py --run $RUNROOT/$r --out $X" > "$X/extract.log" 2>&1 || return 1
    fi
    mkdir -p "$X/msgs_z" "$X/replay"; chmod 777 "$X/replay"
    while IFS=$'\t' read -r sc _; do ln -f "$X"/msgs/clipgt-$sc-*.pkl "$X/msgs_z/" || return 1; done < "$X/zeros.tsv"
    card_free "$g"
    docker run --rm --gpus device=$g -v $C/code/lib/pai_core.py:/app/jev-drive/experiments/alpasim/lib/pai_core.py:ro \
      -v $C/code/lib/pai_driver.py:/app/jev-drive/experiments/alpasim/lib/pai_driver.py:ro -v $C:/col1:ro -v /data:/data -e SH30_TAG=P2H10-F-s0 $IMG \
      python /col1/code/scripts/col1_pai_replay.py --msgs "$X/msgs_z" --out "$X/replay" > "$X/replay.log" 2>&1 || return 1
  fi
  dk "$CODE/scripts/col1_pai.py --x $X --out $X/out" > "$X/pai.log" 2>&1
  [[ -f $X/out/pai_decisions.npz && -f $X/out/pai_cases.csv ]] || { echo "$r: col1_pai.py wrote no decisions"; return 1; }   # the md tables after them may fail
  dk "$H cases --x $X --zeros $X/zeros.tsv --run $r --out $X/out" > "$X/cases.log" 2>&1 || return 1
  dk "$CODE/scripts/col1_pai_clip.py --x $X --dec $X/out/pai_decisions.npz --cases $X/out/cases.csv --out $X/gif" > "$X/clip.log" 2>&1 || return 1
  : > "$OUT/idx/$r.tsv"
  while IFS=$'\t' read -r sc score why _; do
    w=$(awk -F'\t' -v s="$sc" '$1 == s {print $2}' "$X/out/why.tsv"); w=${w:-$why}; f=$(ls "$X"/gif/pai_*_"$sc".gif 2>/dev/null | head -1)
    if [[ -n $f ]]; then cp "$f" "$OUT/gif/${arm}_${l}_${sc}_${w}.gif"; printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$l" "$sc" "$score" "$w" "gif/${arm}_${l}_${sc}_${w}.gif" >> "$OUT/idx/$r.tsv"
    else echo "$r $sc: no gif"; printf '%s\t%s\t%s\t%s\t%s\t\n' "$arm" "$l" "$sc" "$score" "$why" >> "$OUT/idx/$r.tsv"; : > "$OUT/fail/$r.$sc"; fi
  done < "$X/zeros.tsv"
  [[ -n ${KEEP:-} || -n ${PRE:-} ]] || rm -rf "$X/msgs" "$X/msgs_z"        # the messages are ~0.7 GB per run
}
worker() {  # card
  local r
  for r in "${R[@]}"; do
    mkdir "$OUT/claim/$r" 2>/dev/null || continue
    echo "$(date '+%F %T') card $1: $r"
    one "$r" "$1" || { echo "$(date '+%F %T') $r FAILED"; : > "$OUT/fail/$r"; }
  done; }
worker 1 & p1=$!; worker 0 & p0=$!
while kill -0 $p0 2>/dev/null || kill -0 $p1 2>/dev/null; do st "$(ls "$OUT/claim" | wc -l) / ${#R[@]} runs claimed, $(ls "$OUT/gif" | wc -l) gifs"; sleep 60; done
{ printf 'arm\tlist\tscene\tscore\twhy\tgif\n'; cat "$OUT"/idx/*.tsv 2>/dev/null | sort -k2,2 -k1,1 -k3,3; } > "$OUT/index.tsv"
if [[ -n $(ls "$OUT/fail" 2>/dev/null) ]]; then echo "failed: $(ls "$OUT/fail" | tr '\n' ' ')" > "$OUT/ERROR"; st "ERROR $(cat "$OUT/ERROR")"
else date > "$OUT/DONE"; st "done: $(ls "$OUT/gif" | wc -l) gifs of $(($(wc -l < "$OUT/index.tsv") - 1)) zero rollouts, $(du -sh "$OUT/gif" | cut -f1)"; fi
