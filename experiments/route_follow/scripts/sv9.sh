#!/usr/bin/env bash
# SPIN v9 on Tokyo (plans/2026-10-15-spin-v9-prereg.md): PAI, gains 20 (WIZ), serving JEV_VCONT 1.0 + JEV_LEAD 1 as the identities
# (r281/rsg, v2pai/APP1X-L18-s0), one stack per card, RUNPFX v9- (PFX) so no compose name clashes with another lane.
#   sv9.sh smoke               card CARD (0), one after the other: APP1X-L18-s0 switch off on 2 spin-set scenes (plans must equal the stored
#                              identity), then the same 2 scenes with JEV_TRACK=v9 (guard records present); -> $O/smoke_{off,v9}
#   sv9.sh run <arm> <tag> <list> <card> [driver env]   one arm on one card -> $O/<arm>_<tag>, RUN_DONE when finished (start it in tmux)
#   sv9.sh read <arm>          sv9_read.py for both checkpoints -> results/spin_v9/<arm>_{app1x,rsg}.md (Tokyo working tree)
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); repo=$(cd "$here/../../.." && pwd)
D=${DATA_DIR:-/data}; O=$D/runs/route_night/spin_v9; R=$repo/experiments/route_follow/results/spin_v9; S=$repo/experiments/steer_long/results/spin_fix
ID_APP=$D/runs/route_night/v2pai/APP1X-L18-s0; ID_RSG=$D/runs/route_night/r281/rsg
export WIZ=controller.gains.rel_front_steering_angle_weight=20
mkdir -p "$O"
arm() {  # out tag list card extra-env
  mkdir -p "$1"; git -C "$repo" log --oneline -1 > "$1/COMMIT"
  RUNPFX=${PFX:-v9-} CARDS=$4 CHUNK=24 SERVE="-e JEV_VCONT=1.0 -e JEV_LEAD=1 ${5:-}" bash "$repo/experiments/alpasim/scripts/pai_eval.sh" "$1" "$2" "$3"
}
case ${1:?smoke|run|read} in
smoke)
  head -2 "$R/app1x_spin29.txt" > "$O/smoke2.txt"
  rm -rf "$O"/smoke_{off,v9}
  PFX=v9off- arm "$O/smoke_off" APP1X-L18-s0 "$O/smoke2.txt" "${CARD:-0}"
  PFX=v9on- arm "$O/smoke_v9" APP1X-L18-s0 "$O/smoke2.txt" "${CARD:-0}" "-e JEV_TRACK=v9"
  grep -h "serving:" "$O"/smoke_*/runs/*/driver.log
  python3 "$here/sv9_check.py" plans --a "$ID_APP" --b "$O/smoke_off" && echo "SMOKE OFF: identical to the identity"
  python3 "$here/sv9_check.py" plans --a "$O/smoke_off" --b "$O/smoke_v9"
  python3 "$here/sv9_read.py" --label smoke --id "$ID_APP" --arm "$O/smoke_v9" --spin "$O/smoke2.txt" --ctrl "$O/smoke2.txt" --union "$O/smoke2.txt" --md "$O/smoke_v9/read.md" | tail -1
  ;;
run)
  a=$2 t=$3 l=$4 g=$5; x=${6:-}; out=$O/${a}_$t
  arm "$out" "$t" "$l" "$g" "$x"; date > "$out/RUN_DONE"
  ;;
read)
  a=$2
  python3 "$here/sv9_read.py" --label "$a APP1X-L18-s0" --id "$ID_APP" --arm "$O/${a}_APP1X-L18-s0" --spin "$R/app1x_spin29.txt" --ctrl "$R/ctrl_app1x.txt" \
    --ctrl-all "$S/ctrl_all.txt" --union "$R/union90.txt" --md "$R/${a}_app1x.md"
  python3 "$here/sv9_read.py" --label "$a RSG-L18-s0" --id "$ID_RSG" --arm "$O/${a}_RSG-L18-s0" --spin "$S/spin_set.txt" --ctrl "$S/ctrl_all.txt" \
    --union "$R/union90.txt" --md "$R/${a}_rsg.md"
  ;;
esac
