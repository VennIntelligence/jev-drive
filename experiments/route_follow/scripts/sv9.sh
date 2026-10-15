#!/usr/bin/env bash
# SPIN v9 on Tokyo (plans/2026-10-15-spin-v9-prereg.md): PAI, gains 20 (WIZ), serving JEV_VCONT 1.0 + JEV_LEAD 1 as the identities
# (r281/rsg, v2pai/APP1X-L18-s0), one stack per card, RUNPFX v9- (PFX) so no compose name clashes with another lane.
#   sv9.sh smoke               card CARD (0), one after the other: APP1X-L18-s0 switch off on 2 spin-set scenes (plans must equal the stored
#                              identity), then the same 2 scenes with JEV_TRACK=v9 (guard records present); -> $O/smoke_{off,v9}
#   sv9.sh run <arm> <tag> <list> <card> [driver env]   one arm on one card -> $O/<arm>_<tag>, RUN_DONE when finished (start it in tmux)
#   sv9.sh read <arm>          sv9_read.py for both checkpoints -> results/spin_v9/<arm>_{app1x,rsg}.md (Tokyo working tree)
#   sv9.sh ident0              Amendment 4: on the first card free of PAI stacks for 2 min, RSG-L18-s0 on 2 control scenes with JEV_TRACK=0
#                              (plans must equal the stored r281/rsg) and with JEV_TRACK unset (the new default: gate records) -> $O/ident0_{off,default}
#   sv9.sh gateread            the PAI 281 reread of RSG-L18-s0 under the default serving (decision 309), out $V/RSG-L18-s0-gate with the
#                              reference's chunk lists: each card is taken as soon as it has been 2 min free of PAI stacks (the second runs
#                              ident0 first, then joins with JOIN=1 while chunks are unclaimed); then rf_v2pai.py served and rf_v2pai.sh read
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); repo=$(cd "$here/../../.." && pwd)
D=${DATA_DIR:-/data}; O=$D/runs/route_night/spin_v9; R=$repo/experiments/route_follow/results/spin_v9; S=$repo/experiments/steer_long/results/spin_fix
PY=${PY:-/data/envs/op-train/bin/python}     # sv9_read.py needs pyarrow (metrics.parquet event times)
ID_APP=$D/runs/route_night/v2pai/APP1X-L18-s0; ID_RSG=$D/runs/route_night/r281/rsg
export WIZ=controller.gains.rel_front_steering_angle_weight=20
mkdir -p "$O"
arm() {  # out tag list card extra-env
  mkdir -p "$1"; git -C "$repo" log --oneline -1 > "$1/COMMIT"
  RUNPFX=${PFX:-v9-} CARDS=$4 CHUNK=24 SERVE="-e JEV_VCONT=1.0 -e JEV_LEAD=1 ${5:-}" bash "$repo/experiments/alpasim/scripts/pai_eval.sh" "$1" "$2" "$3"
}
isfree() {  # card: < 1000 MiB used for 2 min (12 samples, 10 s apart), i.e. no PAI stack on it
  local n=0 m
  while (( n < 12 )); do
    m=$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits)
    (( m < 1000 )) || return 1; n=$((n + 1)); sleep 10
  done
}
freecard() {  # [exclude] -> the first card free for 2 min; waits
  local g
  while true; do
    for g in 0 1; do [[ $g == "${1:-x}" ]] || ! isfree "$g" || { echo "$g"; return 0; }; done
    sleep 60
  done
}
ident0() {  # card: Amendment 4's bit-identity proof (RSG-L18-s0, 2 control scenes): JEV_TRACK=0 vs r281/rsg, and the unset default
  sed -n 1,2p "$S/ctrl_all.txt" > "$O/ident0.txt"; rm -rf "$O"/ident0_{off,default}
  echo "$(date '+%F %T') ident0 on card $1"
  { PFX=id0off- arm "$O/ident0_off" RSG-L18-s0 "$O/ident0.txt" "$1" "-e JEV_TRACK=0"
    PFX=id0def- arm "$O/ident0_default" RSG-L18-s0 "$O/ident0.txt" "$1"
    grep -h "serving:" "$O"/ident0_*/runs/*/driver.log
    python3 "$here/sv9_check.py" plans --a "$ID_RSG" --b "$O/ident0_off" && echo "IDENT0: JEV_TRACK=0 identical to r281/rsg"
    python3 "$here/sv9_check.py" plans --a "$O/ident0_off" --b "$O/ident0_default"
    python3 - "$O/ident0_default" <<'PY'
import glob, json, sys
n = t = 0
for f in glob.glob(sys.argv[1] + "/runs/*/driver/drive.jsonl"):
    for line in open(f):
        if '"infer": true' in line:
            r = json.loads(line); n += 1; t += r.get("track", {}).get("mode") == "gate"
print(f"default: {t} / {n} decisions with a gate track record")
PY
  } > "$O/ident0.log" 2>&1
}
claimed() { (( $(ls "$1/claims" 2>/dev/null | wc -l) >= $(ls "$1/chunks" 2>/dev/null | wc -l) )) && [[ -d $1/chunks ]]; }
JL="-e JEV_TRACK=gate -e JEV_TRACK_CFG=jlon_v=8"
case ${1:?smoke|run|read|ident0|gateread|jlon|jlon281} in
jlon)
  # Amendment 6: gate + jlon_v=8, RSG-L18-s0 on the 84 (d291 spin set + controls), APP1X-L18-s0 on its 29, behind the gate reread (starts once
  # all of the reread's chunks are claimed), each on a card free for 2 min; APP1X on the other card when it frees, else after RSG on the same
  V=$D/runs/route_night/v2pai; G=$V/RSG-L18-s0-gate
  until claimed "$G"; do sleep 60; done
  cat "$S/spin_set.txt" "$S/ctrl_all.txt" > "$O/s84.txt"
  g=$(freecard); echo "$(date '+%F %T') jlon RSG on card $g"
  ( PFX=jl- arm "$O/jlon_RSG-L18-s0" RSG-L18-s0 "$O/s84.txt" "$g" "$JL"; date > "$O/jlon_RSG-L18-s0/RUN_DONE" ) & p1=$!
  h=$((1 - g)); sleep 150
  while true; do
    if [[ -f $O/jlon_RSG-L18-s0/RUN_DONE ]]; then c=$g; break; fi
    if isfree "$h"; then c=$h; break; fi
    sleep 60
  done
  echo "$(date '+%F %T') jlon APP1X on card $c"
  PFX=jl- arm "$O/jlon_APP1X-L18-s0" APP1X-L18-s0 "$R/app1x_spin29.txt" "$c" "$JL"; date > "$O/jlon_APP1X-L18-s0/RUN_DONE"
  wait $p1
  "$PY" "$here/sv9_read.py" --label "A6 jlon_v=8 RSG-L18-s0 vs v8 (RSG-L18-s0-gate reread)" --id "$G" --arm "$O/jlon_RSG-L18-s0" \
    --spin "$S/spin_set.txt" --ctrl "$S/ctrl_all.txt" --union "$O/s84.txt" --a6 268949d2,09528162 --md "$R/a6_jlon_rsg.md" | tail -n 1
  "$PY" "$here/sv9_read.py" --label "A6 jlon_v=8 APP1X-L18-s0 vs v8 (v8 diagnostic)" --id "$O/v8_APP1X-L18-s0" --arm "$O/jlon_APP1X-L18-s0" \
    --spin "$R/app1x_spin29.txt" --ctrl "$R/app1x_spin29.txt" --union "$R/app1x_spin29.txt" --a6 none --md "$R/a6_jlon_app1x.md" | tail -n 1
  date > "$O/JLON_DONE"
  ;;
jlon281)
  # Amendment 6, only after both jlon reads pass: PAI 281 of RSG-L18-s0 with gate + jlon_v=8, paired vs RSG-L18-s0-gate (same card rules)
  V=$D/runs/route_night/v2pai; J=$V/RSG-L18-s0-jlon8; G=$V/RSG-L18-s0-gate
  mkdir -p "$J/chunks"; ls "$J"/chunks/*.txt >/dev/null 2>&1 || cp "$G"/chunks/*.txt "$J/chunks/"; git -C "$repo" log --oneline -1 > "$J/COMMIT"
  ev() { RUNPFX=jl8- CHUNK=36 WIZ=$WIZ SERVE="-e JEV_VCONT=1.0 -e JEV_LEAD=1 $JL" bash "$repo/experiments/alpasim/scripts/pai_eval.sh" "$J" RSG-L18-s0 "$D/runs/route_night/test281.txt"; }
  g=$(freecard); echo "$(date '+%F %T') jlon281: card $g"; CARDS=$g ev & p1=$!
  h=$((1 - g)); left() { ! claimed "$J"; }
  ( sleep 120; while left; do if isfree "$h" && left; then JOIN=1 CARDS=$h ev; fi; sleep 60; done ) & p2=$!
  wait $p1 $p2
  python3 "$repo/experiments/route_follow/scripts/rf_v2pai.py" served --run "$J" > "$J/served.txt" 2>&1; date > "$J/RUN_DONE"
  REF=$G REFPKL=$G/follow.pkl bash "$repo/experiments/route_follow/scripts/rf_v2pai.sh" read RSG-L18-s0-jlon8
  "$PY" "$here/sv9_read.py" --label "A6 281: jlon_v=8 vs RSG-L18-s0-gate" --id "$G" --arm "$J" --spin "$S/spin_set.txt" --ctrl "$S/ctrl_all.txt" \
    --union "$D/runs/route_night/test281.txt" --md "$R/a6_jlon_281.md" | sed -n 3,8p
  date > "$O/JLON281_DONE"
  ;;
ident0)
  ident0 "$(freecard)"
  ;;
gateread)
  # the reference for every later PAI read (decision 309): the first card free for 2 min starts it at once; the other card, when it is
  # free for 2 min, first runs ident0 (8 min) and then joins while chunks are unclaimed; ident0 runs on the first card at the end if the
  # other card never freed. Never pre-empts another lane (a card is taken only when it holds no stack).
  V=$D/runs/route_night/v2pai; G=$V/RSG-L18-s0-gate; REF=$D/runs/route_night/r281/rsg
  mkdir -p "$G/chunks"; ls "$G"/chunks/*.txt >/dev/null 2>&1 || cp "$REF"/chunks/*.txt "$G/chunks/"; git -C "$repo" log --oneline -1 > "$G/COMMIT"
  ev() { RUNPFX=gate- CHUNK=36 WIZ=$WIZ bash "$repo/experiments/alpasim/scripts/pai_eval.sh" "$G" RSG-L18-s0 "$D/runs/route_night/test281.txt"; }
  g=$(freecard); echo "$(date '+%F %T') gateread: card $g"; CARDS=$g ev & p1=$!
  h=$((1 - g)); left() { (( $(ls "$G/claims" 2>/dev/null | wc -l) < $(ls "$G/chunks" | wc -l) )); }
  ( sleep 120; while left; do
      if isfree "$h" && left; then
        [[ -f $O/IDENT0_DONE ]] || { ident0 "$h"; date > "$O/IDENT0_DONE"; }
        left && { echo "$(date '+%F %T') gateread: card $h joins"; JOIN=1 CARDS=$h ev; }
      fi
      sleep 60; done ) & p2=$!
  wait $p1 $p2
  [[ -f $O/IDENT0_DONE ]] || { ident0 "$g"; date > "$O/IDENT0_DONE"; }
  python3 "$repo/experiments/route_follow/scripts/rf_v2pai.py" served --run "$G" > "$G/served.txt" 2>&1; date > "$G/RUN_DONE"
  bash "$repo/experiments/route_follow/scripts/rf_v2pai.sh" read RSG-L18-s0-gate
  ;;
smoke)
  head -2 "$R/app1x_spin29.txt" > "$O/smoke2.txt"
  rm -rf "$O"/smoke_{off,v9}
  PFX=v9off- arm "$O/smoke_off" APP1X-L18-s0 "$O/smoke2.txt" "${CARD:-0}"
  PFX=v9on- arm "$O/smoke_v9" APP1X-L18-s0 "$O/smoke2.txt" "${CARD:-0}" "-e JEV_TRACK=v9"
  grep -h "serving:" "$O"/smoke_*/runs/*/driver.log
  python3 "$here/sv9_check.py" plans --a "$ID_APP" --b "$O/smoke_off" && echo "SMOKE OFF: identical to the identity"
  python3 "$here/sv9_check.py" plans --a "$O/smoke_off" --b "$O/smoke_v9"
  "$PY" "$here/sv9_read.py" --label smoke --id "$ID_APP" --arm "$O/smoke_v9" --spin "$O/smoke2.txt" --ctrl "$O/smoke2.txt" --union "$O/smoke2.txt" --md "$O/smoke_v9/read.md" | tail -1
  ;;
run)
  a=$2 t=$3 l=$4 g=$5; x=${6:-}; out=$O/${a}_$t
  arm "$out" "$t" "$l" "$g" "$x"; date > "$out/RUN_DONE"
  ;;
read)
  a=$2
  "$PY" "$here/sv9_read.py" --label "$a APP1X-L18-s0" --id "$ID_APP" --arm "$O/${a}_APP1X-L18-s0" --spin "$R/app1x_spin29.txt" --ctrl "$R/ctrl_app1x.txt" \
    --ctrl-all "$S/ctrl_all.txt" --union "$R/union90.txt" --md "$R/${a}_app1x.md"
  "$PY" "$here/sv9_read.py" --label "$a RSG-L18-s0" --id "$ID_RSG" --arm "$O/${a}_RSG-L18-s0" --spin "$S/spin_set.txt" --ctrl "$S/ctrl_all.txt" \
    --union "$R/union90.txt" --md "$R/${a}_rsg.md"
  ;;
esac
