#!/usr/bin/env bash
# PAI2 baseline chain on the GPU box: P2H10-F-s0 and P2H10-F-s1 (current driver, no serving switches) on every chunk of results/pai/chunks as soon as
# its scenes are in the cache. A chunk = one scene list = one simulator launch (the simulator only reproduces for identical lists): a10, b1, b2a, b2b
# are Tokyo's chunks of the 40 (a10 = stage 1 of pai_scenes_40.tsv, b* = COL1's baseline lists), e1..e6 are 10-scene slices of the extension list in
# file order. At most MAXSTACKS simulator jobs of this chain are alive at once; a failed job is resubmitted once by this script, a second failure ends the chain with ERROR.
# STATUS / DONE / ERROR in $DATA_DIR/runs/alpasim/pai2/chain; run dirs $DATA_DIR/runs/alpasim/pai2/base/<chunk>_<s0|s1>.
#   scripts/tmux_run.sh pai2-chain bash experiments/alpasim/scripts/pai2_chain.sh [chunk ...]
set -uo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
P=experiments/alpasim/results/pai; R=$DATA_DIR/runs/alpasim/pai2; K=$R/chain; MAX=${MAXSTACKS:-6}; mkdir -p "$K" "$R/base"; [[ -e $R/base/a10_s0 ]] || ln -s ../par10a "$R/base/a10_s0"; [[ -e $R/base/a10_s1 ]] || ln -s ../base_a10_s1 "$R/base/a10_s1"; rm -f "$K/DONE" "$K/ERROR"
NUREC=${NUREC:-$DATA_DIR/datasets/nurec}; CHUNKS=("$@"); ((${#CHUNKS[@]})) || CHUNKS=(a10 b1 b2a b2b e1 e2 e3 e4 e5 e6)
st() { echo "$(date '+%F %T') $*" | tee "$K/STATUS"; }
declare -A uuid; while IFS=$'\t' read -r s u _; do uuid[$s]=$u; done < <(cat $P/pai_scenes_40.tsv $P/pai_scenes_ext120.tsv | grep -v '^scene_id')
ready() { local s; while read -r s; do [[ -s $NUREC/all-usdzs/${uuid[$s]}.usdz ]] || return 1; done < "$P/chunks/$1.txt"; }
done_() { python3 -c "import json,sys;sys.exit(json.load(open('$R/base/$1/native_summary.json'))['rc'] != 0)" 2>/dev/null; }
failed() { [[ -f $R/base/$1/native_summary.json ]] && ! done_ "$1"; }
declare -A sub tries; todo=(); for c in "${CHUNKS[@]}"; do for t in s0 s1; do todo+=("${c}_$t"); done; done
alive() { local n=0 j; for j in "${!sub[@]}"; do [[ -f $R/base/$j/native_summary.json ]] || n=$((n + 1)); done; echo $n; }
while :; do
  left=0
  for j in "${todo[@]}"; do
    done_ "$j" && continue
    if failed "$j"; then
      (( ${tries[$j]:-0} >= 1 )) && { st "ERROR $j failed twice, see $R/base/$j"; echo "$j" > "$K/ERROR"; exit 1; }
      tries[$j]=1; mv "$R/base/$j/native_summary.json" "$R/base/$j/native_summary.failed.json"; unset "sub[$j]"; st "$j failed, resubmitting"
    fi
    left=$((left + 1)); [[ -n ${sub[$j]:-} ]] && continue
    c=${j%_*}; t=${j#*_}; ready "$c" || continue
    (( $(alive) < MAX )) || continue
    d=$R/base/$j; mkdir -p "$d/pool"
    [[ -e $d/scene_ids.txt && ${tries[$j]:-0} == 0 ]] && { sub[$j]=earlier; continue; }   # a10_s0 / a10_s1 were submitted by hand (par10a, base_a10_s1)
    id=$(.venv/bin/python -m jevdrive.cl submit --no-check --name "pai2-base-$c-$t" --vram 24 --cpu 6 --ram 40 --log-dir "$d/pool" -- \
      env CONC=4 SH30_TAG=P2H10-F-$t bash experiments/alpasim/scripts/pai_native.sh "$d" "$P/chunks/$c.txt" 2>&1 | tail -1)
    sub[$j]=$id; st "submitted $j as $id"
  done
  ((left == 0)) && break
  sleep 60
done
date > "$K/DONE"; st "done"
