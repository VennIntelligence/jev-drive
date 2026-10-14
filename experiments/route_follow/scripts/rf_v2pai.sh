#!/usr/bin/env bash
# Route v2 on PAI 281 (Tokyo), paired against the stored RSG-L18-s0 read, and the turn-gain switch diagnostic. Serving = the reference
# /data/runs/route_night/r281/rsg exactly (route_night/chain2.sh): pai_eval.sh, wizard controller.gains.rel_front_steering_angle_weight=20,
# SERVE "-e JEV_VCONT=1.0 -e JEV_LEAD=1", CONC 4, one stack per card on both cards, the reference's own 8 chunk lists (36 scenes).
#   rf_v2pai.sh run <tag>      copy <tag>'s ckpt-final.pt + route_mem.json from the box (sha256 checked; skipped when Tokyo already has the
#                              same files), launch the 281 read in tmux jev:v2-<tag> -> $V/<tag>; after it: rf_v2pai.py served -> served.txt
#   rf_v2pai.sh read <tag>     rf_pai281.py (all 281 + exam 72 paired vs rsg), rf_pai_follow extract (docker, CPU), rf_v2pai.py turns / table
#                              on the 36 strict turns + 72 exam scenes -> $V/<tag>/read.md and experiments/route_follow/results/pai281/<tag>.md
#                              (Tokyo working tree: copy it to the Mac and commit there)
#   rf_v2pai.sh tg <g>         turn-gain diagnostic (plans/2026-10-15-turn-gain-switch-prereg.md): RSG-L18-s0 + JEV_TURN_GAIN=<g> on
#                              results/turn_gain/scenes47.txt (2 chunks, both cards) -> $D/runs/route_night/turn_gain/g<g>, then tgcheck,
#                              extract, turns, table -> results/turn_gain/g<g>.md (Tokyo working tree)
#   rf_v2pai.sh chain <tag> <next>   no idle card between two 281 reads: copy <next>'s checkpoint, then as each card finishes <tag>'s last
#                              chunk (all of <tag>'s chunks claimed, the card below 1 GB for 1 min) start <next> on it (the first freed card
#                              owns the run: STATUS / DONE; the second joins with JOIN=1); served check -> RUN_DONE as in run. Foreground:
#                              start it in tmux.
#   rf_v2pai.sh sc <tag> <clip:<m> | zero> <scene list>   the ordered-memory scalar diagnostic (route v2 prereg Amendment 16, JEV_RM_SC) on
#                              CARDS (one card), out $V/sc_<tag>_<mode>, then rf_v2pai.py spins vs r281/rsg -> spins.md
#   rf_v2pai.sh tgsmoke <g>    the same on the first 2 scenes of scenes47 (card 0), tgcheck only
# Env: DRY=<n> (run: the first n scenes of the reference chunk c000 in chunks of n / 2, out $V/dry_<tag>, no tmux: foreground),
#   V (/data/runs/route_night/v2pai), BOX (autodl), BOXD (/root/autodl-tmp/ujs), CARDS ("0 1").
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); repo=$(cd "$here/../../.." && pwd)
D=${DATA_DIR:-/data}; V=${V:-$D/runs/route_night/v2pai}; REF=$D/runs/route_night/r281/rsg; BOX=${BOX:-autodl}; BOXD=${BOXD:-/root/autodl-tmp/ujs}
RES=$repo/experiments/route_follow/results; EXAM=$repo/experiments/alpasim/results/pai_exam
export WIZ=controller.gains.rel_front_steering_angle_weight=20
cmd=${1:?run|read|tg|tgsmoke}; arg=${2:?tag or gain}
mkdir -p "$V"

ckpt() {  # tag: the checkpoint from the box, sha256 equal on both sides
  local t=$1 d=$D/runs/op_parity/runs/$1 f a b
  mkdir -p "$d"
  for f in ckpt-final.pt route_mem.json; do
    a=$(ssh "$BOX" "sha256sum $BOXD/runs/op_parity/runs/$t/$f" 2>/dev/null | cut -d' ' -f1)
    [[ -n $a ]] || { echo "no $f of $t on $BOX" >&2; return 1; }
    b=$(sha256sum "$d/$f" 2>/dev/null | cut -d' ' -f1)
    [[ $a == "$b" ]] || rsync -a "$BOX:$BOXD/runs/op_parity/runs/$t/$f" "$d/" 2>/dev/null
    b=$(sha256sum "$d/$f" | cut -d' ' -f1)
    [[ $a == "$b" ]] || { echo "$t/$f: sha256 differs after the copy" >&2; return 1; }
  done
  echo "$t: checkpoint on Tokyo, sha256 $(sha256sum "$d/ckpt-final.pt" | cut -c1-12) = box"
}

follow() {  # out pkl, LABEL=root ...: rf_pai_follow extract (alpasim-base, CPU)
  WORKERS=${WORKERS:-8} CPUS=${CPUS:-8} bash "$here/rf_pai_follow.sh" "$@"
}

case $cmd in
run)
  t=$arg; ckpt "$t" || exit 1
  if [[ -n ${DRY:-} ]]; then
    O=$V/dry_$t; rm -rf "$O"; mkdir -p "$O"; git -C "$repo" log --oneline -1 > "$O/COMMIT"; head -n "$DRY" "$REF/chunks/c000.txt" > "$O/scenes.txt"
    CHUNK=$(( (DRY + 1) / 2 )) bash "$repo/experiments/alpasim/scripts/pai_eval.sh" "$O" "$t" "$O/scenes.txt"
    python3 "$here/rf_v2pai.py" served --run "$O" | tee "$O/served.txt"
    exit 0
  fi
  O=$V/$t; mkdir -p "$O/chunks"
  ls "$O"/chunks/*.txt >/dev/null 2>&1 || cp "$REF"/chunks/*.txt "$O/chunks/"          # the reference's own chunk lists
  git -C "$repo" log --oneline -1 > "$O/COMMIT"
  # the tmux window does not inherit this script's environment: WIZ (gains 20) and CARDS go on its command line (the first RV2a-L18-s0 run
  # was launched without them and drove at the default gains: route v2 prereg Amendment 18)
  tmux new-window -d -t jev -n "v2-$t" "bash -c 'WIZ=$WIZ CARDS=\"${CARDS:-0 1}\" CHUNK=36 bash $repo/experiments/alpasim/scripts/pai_eval.sh $O $t $D/runs/route_night/test281.txt; \
python3 $here/rf_v2pai.py served --run $O > $O/served.txt 2>&1; date > $O/RUN_DONE'"
  echo "launched $t -> $O (tmux jev:v2-$t); state $O/STATUS, done when $O/RUN_DONE exists"
  ;;
read)
  t=$arg; O=$V/$t; l=${t,,}
  [[ -f $O/DONE ]] || { echo "$O: no DONE (pai_eval unfinished or failed: $(cat "$O/STATUS"))" >&2; exit 1; }
  python3 "$here/rf_pai281.py" --rows rsg="$REF" "$l=$O" --ref rsg --lists all281=$D/runs/route_night/test281.txt exam72="$EXAM/hard_exam.txt" control75="$EXAM/control.txt" --vs rsg --out "$O/read281.md"
  [[ -f $O/follow.pkl ]] || follow "$O/follow.pkl" "$l=$O" || exit 1
  cat "$RES/pai281/strict36.txt" "$EXAM/hard_exam.txt" | awk '!s[$1]++' > "$O/turn_scenes.txt"
  python3 "$here/rf_v2pai.py" turns --pkl rsg=$D/runs/route_night/pai_follow/r281.pkl "$l=$O/follow.pkl" --scenes "$O/turn_scenes.txt" --out "$O/turns.json" || exit 1
  python3 "$here/rf_v2pai.py" table --turns "$O/turns.json" --a rsg --b "$l" --strict "$RES/pai281/strict36.txt" --exam "$EXAM/hard_exam.txt" --md "$O/turns.md" || exit 1
  { echo "# PAI 281: $t vs RSG-L18-s0 (stored read r281/rsg), gains 20, JEV_VCONT 1.0, JEV_LEAD 1"
    echo; echo "Run \`$O\` ($(cat "$O/COMMIT" 2>/dev/null)), reference \`$REF\`; rf_v2pai.sh read. Served memory check (rf_v2pai.py served):"
    echo; tail -2 "$O/served.txt" | sed 's/^/- /'
    echo; echo "## All 281 and the exam / control lists (rf_pai281.py; turn proxy = the 20 reference turn scenes)"; echo; cat "$O/read281.md"
    echo; echo "## Strict turns (36) and exam scenes (72): rf_pai_follow's turn rule on each label's own route (rf_v2pai.py turns / table)"; echo
    sed 's/^# /### /' "$O/turns.md"; } > "$O/read.md"
  cp "$O/read.md" "$RES/pai281/$t.md"; cp "$O/turns.json" "$RES/pai281/$t.turns.json"
  echo "-> $RES/pai281/$t.md"
  ;;
sc)
  t=$arg; m=${3:?clip:<m> | zero}; L=${4:?scene list}; O=$V/sc_${t}_${m/:/}; mkdir -p "$O"; git -C "$repo" log --oneline -1 > "$O/COMMIT"
  SERVE="-e JEV_VCONT=1.0 -e JEV_LEAD=1 -e JEV_RM_SC=$m" RUNPFX=sc${m:0:1}- CHUNK=36 CARDS=${CARDS:-0} \
    bash "$repo/experiments/alpasim/scripts/pai_eval.sh" "$O" "$t" "$L"
  python3 "$here/rf_v2pai.py" spins --a rsg="$REF" --b "${m/:/}=$O" --out "$O/spins.md" > /dev/null
  date > "$O/SC_DONE"
  ;;
chain)
  t=$arg; n=${3:?next tag}; P=$V/$t; O=$V/$n; ckpt "$n" || exit 1
  mkdir -p "$O/chunks"; ls "$O"/chunks/*.txt >/dev/null 2>&1 || cp "$REF"/chunks/*.txt "$O/chunks/"
  git -C "$repo" log --oneline -1 > "$O/COMMIT"
  free() { (( $(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits) < 1000 )); }
  started=(); pids=()
  while (( ${#started[@]} < 2 )); do
    if (( $(ls "$P/claims" 2>/dev/null | wc -l) >= $(ls "$P/chunks" | wc -l) )); then
      for g in 0 1; do
        [[ " ${started[*]} " == *" $g "* ]] && continue
        free "$g" && sleep 60 && free "$g" || continue
        echo "$(date '+%F %T') card $g free: $n starts on it"
        if (( ${#started[@]} == 0 )); then CHUNK=36 CARDS=$g bash "$repo/experiments/alpasim/scripts/pai_eval.sh" "$O" "$n" "$D/runs/route_night/test281.txt" & pids+=($!)
        else sleep 30; JOIN=1 CHUNK=36 CARDS=$g bash "$repo/experiments/alpasim/scripts/pai_eval.sh" "$O" "$n" "$D/runs/route_night/test281.txt" & pids+=($!); fi
        started+=("$g")
      done
    fi
    sleep 30
  done
  wait "${pids[@]}"
  python3 "$here/rf_v2pai.py" served --run "$O" > "$O/served.txt" 2>&1; date > "$O/RUN_DONE"
  ;;
tg|tgsmoke)
  g=$arg; t=RSG-L18-s0; L=$RES/turn_gain/scenes47.txt; O=$D/runs/route_night/turn_gain/g$g
  [[ $cmd == tgsmoke ]] && { O=$O-smoke; rm -rf "$O"; mkdir -p "$O"; head -2 "$L" > "$O/scenes.txt"; L=$O/scenes.txt; }
  mkdir -p "$O"; git -C "$repo" log --oneline -1 > "$O/COMMIT"
  SERVE="-e JEV_VCONT=1.0 -e JEV_LEAD=1 -e JEV_TURN_GAIN=$g" RUNPFX=tg- CHUNK=24 CARDS=${CARDS:-$([[ $cmd == tgsmoke ]] && echo 0 || echo "0 1")} \
    bash "$repo/experiments/alpasim/scripts/pai_eval.sh" "$O" "$t" "$L"
  python3 "$here/rf_v2pai.py" tgcheck --run "$O" --ref "$REF" | tee "$O/tgcheck.txt"
  [[ $cmd == tgsmoke ]] && exit 0
  [[ -f $O/DONE ]] || { echo "$O: no DONE" >&2; exit 1; }
  follow "$O/follow.pkl" "g$g=$O" || exit 1
  python3 "$here/rf_v2pai.py" turns --pkl rsg=$D/runs/route_night/pai_follow/r281.pkl "g$g=$O/follow.pkl" --scenes "$RES/turn_gain/scenes47.txt" --tg "g$g=$O" --out "$O/turns.json" || exit 1
  python3 "$here/rf_v2pai.py" table --turns "$O/turns.json" --a rsg --b "g$g" --strict "$RES/pai281/strict36.txt" --exam "$EXAM/exam_turn26.txt" --md "$O/table.md" || exit 1
  cp "$O/table.md" "$RES/turn_gain/g$g.md"; cp "$O/turns.json" "$RES/turn_gain/g$g.turns.json"; tail -3 "$O/tgcheck.txt" > "$RES/turn_gain/g$g.tgcheck.txt"
  date > "$O/READ_DONE"; echo "-> $RES/turn_gain/g$g.md"
  ;;
*) echo "unknown command $cmd" >&2; exit 2 ;;
esac
