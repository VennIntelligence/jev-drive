#!/usr/bin/env bash
# INTENT HOLD serve-side diagnostic on PAI (plans/2026-10-15-intent-hold-prereg.md; lib/intent_hold.py, rf_intent.py). Tokyo. Serving = the
# reference /data/runs/route_night/r281/rsg (RSG-L18-s0, gains 20, JEV_VCONT 1.0, JEV_LEAD 1, CONC 4) plus JEV_INTENT_HOLD=1 JEV_IH_A=<a>.
#   rf_intent.sh chain     wait for a free card (no compute process, < 1 GB, no live pai_eval.sh whose log names it, for 2 min), then on it:
#                          identity re-run without the switch on 2 scenes (rf_p4.py s1 vs r281/rsg), smoke a 0.3 on the same 2 scenes
#                          (rf_intent.py ihcheck), and only when both are clean the a 0.3 run and the a 0.15 run on the scene list (one
#                          stack, back to back). Foreground: start it in tmux. State in $O/CHAIN.log; STOP on a failed check.
#   rf_intent.sh resume    (CARD=<g>) wait for $O/smoke to end, then the checks and the a runs as in chain (prereg Amendment 1)
#   rf_intent.sh read <a>  follow extract, rf_v2pai.py turns / table, spins, rf_intent.py extract (identity + run) -> $O/a<a>/ and the small
#                          outputs to experiments/route_follow/results/intent_hold/ (Tokyo working tree: copy to the Mac, commit there)
# Env: O (/data/runs/route_night/intent_hold), CARD (skip the wait: use this card).
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); repo=$(cd "$here/../../.." && pwd)
D=${DATA_DIR:-/data}; O=${O:-$D/runs/route_night/intent_hold}; REF=$D/runs/route_night/r281/rsg; T=RSG-L18-s0
RES=$repo/experiments/route_follow/results; L=$RES/intent_hold/scenes49.txt; EXAM=$repo/experiments/alpasim/results/pai_exam
export WIZ=controller.gains.rel_front_steering_angle_weight=20
mkdir -p "$O"
log() { echo "$(date '+%F %T') $*" | tee -a "$O/CHAIN.log"; }

busy() {  # card $1 in use: >= 1 GB, a compute process, or a live pai_eval.sh whose log names the card
  local g=$1 pid root
  (( $(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits) >= 1000 )) && return 0
  [[ -n $(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader) ]] && return 0
  for pid in $(pgrep -f "alpasim/scripts/pai_eval.sh"); do
    root=$(tr '\0' '\n' < /proc/$pid/cmdline 2>/dev/null | sed -n 3p)
    [[ -n $root ]] && grep -o "cards [0-9 ]*[,:]" "$root/log.txt" 2>/dev/null | tr -dc '0-9 \n' | grep -qw "$g" && return 0
  done
  return 1
}

waitcard() {  # -> a card free for 120 s on stdout
  local -A free=()
  while true; do
    for g in 0 1; do
      if busy "$g"; then free[$g]=0; else free[$g]=$(( ${free[$g]:-0} + 10 )); fi
      (( ${free[$g]} >= 120 )) && { echo "$g"; return; }
    done
    sleep 10
  done
}

# JEV_TRACK=0 pinned (prereg Amendment 1): decision 309 made `gate` the driver default; the identity r281/rsg is the no-gate path
serve() { echo "-e JEV_VCONT=1.0 -e JEV_LEAD=1 -e JEV_TRACK=0${1:+ -e JEV_INTENT_HOLD=1 -e JEV_IH_A=$1}"; }

runs() {  # card $1: the identity / smoke checks, then a 0.3 and a 0.15 back to back
  local g=$1 a R
  python3 "$here/rf_p4.py" s1 --run "$O/id" --ref "$REF" > "$O/id/s1.txt" 2>&1
  python3 "$here/rf_intent.py" ihcheck --run "$O/smoke" --ref "$REF" > "$O/smoke/ihcheck.txt" 2>&1
  log "identity: $(tail -1 "$O/id/s1.txt")"; log "smoke: $(tail -1 "$O/smoke/ihcheck.txt")"
  grep -q "S1: 2 / 2" "$O/id/s1.txt" || { log "STOP: identity re-run not bit-identical"; touch "$O/STOP"; exit 1; }
  [[ -f $O/smoke/DONE ]] && grep -q "max |diff| 0.0e+00$" "$O/smoke/ihcheck.txt" || { log "STOP: smoke failed (no DONE or a pre-hold plan differs)"; touch "$O/STOP"; exit 1; }
  for a in 0.3 0.15; do
    R=$O/a$a; mkdir -p "$R"; git -C "$repo" log --oneline -1 > "$R/COMMIT"
    log "a $a: $(wc -l < "$L") scenes on card $g, serve $(serve $a)"
    SERVE="$(serve $a)" RUNPFX=ih${a#0.}- CHUNK=25 CARDS=$g bash "$repo/experiments/alpasim/scripts/pai_eval.sh" "$R" "$T" "$L" > "$R/eval.out" 2>&1
    python3 "$here/rf_intent.py" ihcheck --run "$R" --ref "$REF" > "$R/ihcheck.txt" 2>&1
    log "a $a done: $(tail -1 "$R/STATUS" 2>/dev/null); $(tail -1 "$R/ihcheck.txt")"
  done
  date > "$O/CHAIN_DONE"
}

case ${1:?chain|resume|read} in
chain)
  g=${CARD:-$(log "waiting for a card" >&2; waitcard)}
  log "card $g free: identity + smoke start ($(git -C "$repo" log --oneline -1 | cut -c1-60))"
  head -2 "$L" > "$O/smoke2.txt"
  for x in id smoke; do
    R=$O/$x; rm -rf "$R"; mkdir -p "$R"; git -C "$repo" log --oneline -1 > "$R/COMMIT"
    SERVE="$(serve $([[ $x == smoke ]] && echo 0.3))" RUNPFX=ih${x:0:1}- CHUNK=2 CARDS=$g \
      bash "$repo/experiments/alpasim/scripts/pai_eval.sh" "$R" "$T" "$O/smoke2.txt" > "$R/eval.out" 2>&1
  done
  runs "$g"
  ;;
resume)  # Amendment 1: after the smoke of an interrupted chain (card CARD), as soon as it ends: the checks and the a runs, no idle gap
  g=${CARD:?CARD}
  until [[ -f $O/smoke/DONE || -f $O/smoke/ERROR ]]; do sleep 5; done
  log "resume on card $g ($(git -C "$repo" log --oneline -1 | cut -c1-60))"
  runs "$g"
  ;;
read)
  a=${2:?a}; R=$O/a$a; l=ih$a; W=$RES/intent_hold
  [[ -f $R/DONE ]] || { echo "$R: no DONE" >&2; exit 1; }
  [[ -f $R/follow.pkl ]] || WORKERS=${WORKERS:-8} CPUS=${CPUS:-8} bash "$here/rf_pai_follow.sh" "$R/follow.pkl" "$l=$R" || exit 1
  python3 "$here/rf_v2pai.py" turns --pkl rsg=$D/runs/route_night/pai_follow/r281.pkl "$l=$R/follow.pkl" --scenes "$L" --tg "$l=$R" --out "$R/turns.json" || exit 1
  python3 "$here/rf_v2pai.py" table --turns "$R/turns.json" --a rsg --b "$l" --strict "$RES/pai281/strict36.txt" --exam "$EXAM/exam_turn26.txt" --md "$R/table.md" || exit 1
  python3 -c "
import json, sys; sys.path.insert(0, '$here'); import rf_pai281 as P
S = {x.split()[0] for x in open('$L') if x.strip()}
a, b = P.read('$REF'), P.read('$R')
print(json.dumps({'rsg': sum(bool(a[s]['spin']) for s in S if s in a), '$l': sum(bool(b[s]['spin']) for s in S if s in b),
                  'scenes_rsg': sorted(s[7:15] for s in S if s in a and a[s]['spin']), 'scenes_$l': sorted(s[7:15] for s in S if s in b and b[s]['spin'])}))
" > "$R/spins.json" || exit 1
  [[ -f $R/creep.pkl ]] || python3 "$here/rf_intent.py" extract --set "rsg=$D/runs/route_night/pai_follow/r281.pkl:rsg:$REF" "$l=$R/follow.pkl:$l:$R" \
    --scenes "$L" --out "$R/creep.pkl" || exit 1
  mkdir -p "$W"; cp "$R/table.md" "$W/a$a.table.md"; cp "$R/table.counts.json" "$W/a$a.counts.json"; cp "$R/turns.json" "$W/a$a.turns.json"
  cp "$R/spins.json" "$W/a$a.spins.json"; tail -1 "$R/ihcheck.txt" > "$W/a$a.ihcheck.txt"
  date > "$R/READ_DONE"; echo "-> $W/a$a.*; creep pkl $R/creep.pkl (copy to the Mac: rf_intent.py creep / lines / fig)"
  ;;
*) sed -n '2,12p' "$0"; exit 2 ;;
esac
