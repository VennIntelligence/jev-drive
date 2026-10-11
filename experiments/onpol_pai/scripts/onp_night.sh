#!/usr/bin/env bash
# ONPOL-PAI training stages on the Tokyo box, one self-advancing stage per call (tmux session jev, window onp-tr*). A stage never starts
# the next one: the main session reads its gates first.
#   onp_night.sh cache [card]          row cache of pass pol (and pass log if it has row files; incremental), then G-id    ~5 min per pass
#                                      (the registered rule is printed; the stage stops on the working line of the pre-registration's
#                                      deviation: wrong checkpoint >= 0.9 over, largest difference <= 0.25 m, <= 1 % over max(0.03 m, 4 ulp))
#   onp_night.sh check [card|cpu]      onp_train.py check (log consistency, gradient directions, operators)               ~2 min
#   onp_night.sh smoke "<cards>"       60 steps of SELF, IMIT, BOTH, FT (tags P2H10S-*-T-s0, overwritten)                   ~4 min each
#   onp_night.sh pilot "<cards>"       3 000 steps of SELF, IMIT, BOTH (tags -P-), then the held read                     ~10 min each
#   onp_night.sh full  "<cards>"       cache + G-id of pass log as its row files stand at that moment, then 10 000 steps of SELF, IMIT,
#                                      BOTH, FT (tags -F-), then the held read; needs <root>/FULL_GO, which the main session creates
#                                      after reading the pilot                                                           ~26 min each
#   onp_night.sh held  [card] <tag>..  the held read of any tags -> <root>/held/held.{md,json}
#   onp_night.sh gload <tag> [card]    G-load: the checkpoint through sh30_core.Core + pai_core.plan inside the driver image
#   tmux new-window -d -t jev -n onp-tr-pilot "bash experiments/onpol_pai/scripts/onp_night.sh pilot '0 1'"
# Cards: every card walks the arm list in order and takes the next unclaimed arm, so two cards run SELF and IMIT at once, then BOTH and
# FT. A card is used only if <lane>/GPU<card>_FREE exists (the rollouts own the cards otherwise). An arm that ends in an error (a
# non-finite loss stops the trainer) gets <stage>/<ARM>/ERROR and the other arms go on; the stage then ends in ERROR naming it.
# Env: DATA_DIR (/data), PY, LANE ($DATA_DIR/runs/onpol_pai), ROOT ($LANE/train), INIT (P2H10S-F-s0), SEED (0), ARMS (stage default),
#   RECIPE (the S flags), EXTRA (more onp_train.py train flags, e.g. "--on-log 0"), IMG (driver image of gload).
# Layout: $ROOT/<stage>/{STATUS, DONE | ERROR, log.txt, <ARM>/{log.txt, DONE | ERROR}}; checkpoints $DATA_DIR/runs/op_parity/runs/<tag>/.
set -uo pipefail
cd "$(dirname "$0")/../../.."
export DATA_DIR=${DATA_DIR:-/data}; PY=${PY:-$DATA_DIR/envs/op-train/bin/python}
LANE=${LANE:-$DATA_DIR/runs/onpol_pai}; ROOT=${ROOT:-$LANE/train}; INIT=${INIT:-P2H10S-F-s0}; SEED=${SEED:-0}
RECIPE=${RECIPE:---ho ot1:4,yr1:4,bd4:5 --ho-w 3 --ho-excl navsim/body1-val-logs --agent-lam 10 --shape --label-bank --compile}
IMG=${IMG:-jev-alpasim:p2h10-f-s0-6f3d05d1}
T=experiments/onpol_pai/scripts/onp_train.py; C=experiments/onpol_pai/scripts/onp_cache.py
stage=${1:?stage}; shift
R=$ROOT/$stage; mkdir -p "$R"; rm -f "$R"/{DONE,ERROR}; exec > >(tee -a "$R/log.txt") 2>&1
st() { echo "$(date '+%F %T') $stage: $*" | tee "$R/STATUS"; }
die() { st "ERROR $*"; echo "$*" > "$R/ERROR"; exit 1; }
free() { local g; for g in "$@"; do [[ $g == cpu || -f $LANE/GPU${g}_FREE ]] || die "card $g is not free ($LANE/GPU${g}_FREE missing)"; done; }
tag() {  # arm, kind (T | P | F)
  case $1 in FT) echo "P2H10S-FT-$2-s$SEED";; *) echo "P2H10S-$1-ON-$2-s$SEED";; esac; }
arm_flags() { case $1 in SELF) echo "--on-arm self";; IMIT) echo "--on-arm imit";; BOTH) echo "--on-arm both";; FT) echo "--on-rows 0 --on-eval";; *) return 1;; esac; }
train() {  # cards, kind, steps, extra flags: every card takes unclaimed arms of $ARMS in order
  local cards=$1 kind=$2 steps=$3 more=$4 g pids=()
  rm -rf "$R/claims"; mkdir "$R/claims"
  for g in $cards; do
    ( for a in $ARMS; do
        O=$R/$a; t=$(tag "$a" "$kind")
        mkdir "$R/claims/$a" 2>/dev/null || continue
        [[ $kind != T && -f $O/DONE && -f $DATA_DIR/runs/op_parity/runs/$t/ckpt-final.pt ]] && { echo "$(date '+%F %T') card $g: $a ($t) already done"; continue; }
        mkdir -p "$O"; rm -f "$O"/{DONE,ERROR}; echo "$(date '+%F %T') card $g: $a -> $t, $steps steps"
        # shellcheck disable=SC2086
        if CUDA_VISIBLE_DEVICES=$g "$PY" "$T" train --tag "$t" --init "$INIT" --seed "$SEED" --steps "$steps" $(arm_flags "$a") $RECIPE $more ${EXTRA:-} > "$O/log.txt" 2>&1
        then date > "$O/DONE"; echo "$(date '+%F %T') card $g: $a done: $(grep -o 'dev @ .*' "$O/log.txt" | tail -1)"
        else grep -m1 -E 'non-finite|Error|error' "$O/log.txt" | cut -c1-300 > "$O/ERROR"; echo "$(date '+%F %T') card $g: $a ERROR: $(cat "$O/ERROR")"; fi
      done ) & pids+=($!); sleep 5
  done
  wait "${pids[@]}"
}
held() {  # card, name, tags...
  local g=$1 n=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g "$PY" "$T" held --name "$n" --out "$ROOT/held" --tags "$@"
}
finish() {  # kind: held read of the parent and of the arms that finished; stage DONE only if every arm did
  local kind=$1 g=$2 ok=() bad=() a
  for a in $ARMS; do if [[ -f $R/$a/DONE ]]; then ok+=("$(tag "$a" "$kind")"); else bad+=("$a"); fi; done
  if (( ${#ok[@]} )); then st "held read of ${ok[*]}"; held "$g" "$stage" "$INIT" "${ok[@]}" || bad+=("held-read"); fi
  (( ${#bad[@]} )) && die "failed: ${bad[*]} (see $R/<ARM>/ERROR)"
  date > "$R/DONE"; st "done: ${ok[*]}; held read $ROOT/held/$stage.md"
}
cache_pass() {  # pass, card: incremental cache build of the pass's row files, then G-id (registered rule: printed, not a stop; working line: a stop)
  local p=$1 g=$2 rc
  ls "$LANE/rows/$p"/*.npz >/dev/null 2>&1 || { echo "no row file of pass $p: skipped"; return 0; }
  st "cache of pass $p on card $g"
  CUDA_VISIBLE_DEVICES=$g "$PY" "$C" build --pass "$p" || die "cache build of pass $p"
  st "G-id of pass $p"
  CUDA_VISIBLE_DEVICES=$g "$PY" "$C" gid --pass "$p" --tag "$INIT" --control "${CONTROL:-P2H10S-F-s1}" > "$R/gid-$p.out" 2>&1; echo "registered G-id of pass $p: $(grep '^Gate' "$ROOT/cache/gid/$p-$INIT.md")"
  CUDA_VISIBLE_DEVICES=$g "$PY" "$C" gid-ulp --pass "$p" --tag "$INIT" --control "${CONTROL:-P2H10S-F-s1}" > "$R/gid-ulp-$p.out" 2>&1; rc=$?
  grep '^Working line' "$ROOT/cache/gid/$p-$INIT-ulp.md" || die "G-id of pass $p did not run: $R/gid-ulp-$p.out"
  (( rc == 0 )) || die "G-id working line of pass $p failed: $ROOT/cache/gid/$p-$INIT-ulp.md"
}
case $stage in
  cache)
    g=${1:-1}; free "$g"
    for p in pol log; do cache_pass "$p" "$g"; done
    date > "$R/DONE"; st "done: $(grep -h 'Working line:' "$ROOT"/cache/gid/*-"$INIT"-ulp.md | tr '\n' ' ')";;
  check)
    g=${1:-1}; free "$g"; st "running on $g"
    if [[ $g == cpu ]]; then CUDA_VISIBLE_DEVICES= "$PY" "$T" check --device cpu > "$R/check.out" 2>&1; else CUDA_VISIBLE_DEVICES=$g "$PY" "$T" check > "$R/check.out" 2>&1; fi || die "check failed: $R/check.out"
    date > "$R/DONE"; st "done: $ROOT/check/check.json: $(grep -A4 '"pass"' "$ROOT/check/check.json" | tr -d '\n ')";;
  smoke)
    cards=${1:-1}; free $cards; ARMS=${ARMS:-SELF IMIT BOTH FT}; st "arms $ARMS on cards $cards"
    train "$cards" T "${STEPS:-60}" "--force --eval-every 60 ${SMOKE_DATA:-}"
    bad=(); for a in $ARMS; do [[ -f $R/$a/DONE ]] || bad+=("$a"); done
    (( ${#bad[@]} )) && die "failed: ${bad[*]}"
    date > "$R/DONE"; st "done: $(for a in $ARMS; do echo "$a $(grep -o '[0-9.]* it/s, [0-9.]* GB' "$R/$a/log.txt" | tail -1);"; done | tr '\n' ' ')";;
  pilot)
    cards=${1:-1}; free $cards; ARMS=${ARMS:-SELF IMIT BOTH}; st "arms $ARMS, 3000 steps, cards $cards"
    train "$cards" P 3000 ""; finish P "${cards%% *}";;
  full)
    cards=${1:-0 1}; free $cards; ARMS=${ARMS:-SELF IMIT BOTH FT}
    [[ -f $ROOT/FULL_GO ]] || die "$ROOT/FULL_GO missing: the main session reads the pilot gates and creates it; this stage does not start by itself"
    cache_pass log "${cards%% *}"                             # whatever rows/log holds now (only new scenes are encoded); no pass log: all 16 rows from pol
    st "arms $ARMS, 10000 steps, cards $cards"
    train "$cards" F 10000 ""; finish F "${cards%% *}";;
  held)
    g=${1:-1}; free "$g"; shift; st "tags $*"
    held "$g" "${NAME:-held}" "$@" || die "held read"
    date > "$R/DONE"; st "done: $ROOT/held/${NAME:-held}.md";;
  gload)
    t=${1:?tag}; g=${2:-1}; free "$g"; ck=$DATA_DIR/runs/op_parity/runs/$t; lib=$PWD/experiments/alpasim/lib
    [[ -f $ck/ckpt-final.pt ]] || die "no checkpoint $ck"
    st "$t through $IMG on card $g"
    docker run --rm --init --cap-drop ALL --security-opt no-new-privileges:true --read-only --network none --gpus "device=$g" \
      --tmpfs /tmp:rw,nosuid,nodev,size=2g,mode=1777 --tmpfs /run:rw,nosuid,nodev,size=64m,mode=0755 \
      -v "$lib/pai_core.py:/app/jev-drive/experiments/alpasim/lib/pai_core.py:ro" -v "$lib/pai_driver.py:/app/jev-drive/experiments/alpasim/lib/pai_driver.py:ro" \
      -v "$lib/serve_fix.py:/app/jev-drive/experiments/alpasim/lib/serve_fix.py:ro" \
      -v "$PWD/jevdrive/openpilot/lead_long.py:/app/jev-drive/jevdrive/openpilot/lead_long.py:ro" \
      -v "$ck:/app/data/runs/op_parity/runs/$t:ro" -e "SH30_TAG=$t" -e JEV_VCONT=1.0 -e JEV_LEAD=1 "$IMG" python -c '
import os, sys
import numpy as np
sys.path.insert(0, "/app/jev-drive/experiments/alpasim/lib")
import pai_driver as D                                    # the served driver module: its imports are the gate too
from pai_core import C, plan
core = C.Core(os.environ["SH30_TAG"])
core.lead_out, core.base_out = D.FX.NEED_LEAD, D.FX.BASE > 0
z = np.zeros
o = plan(core, z((8,) + C.FRAME, np.uint8), np.ones(8, bool), z((4, 3)), z((4, 2)), z(2), np.array([0, 1, 0, 0]), [1.7, 0.0, 1.5])
assert np.isfinite(o["poses"]).all() and o["poses"].shape == (8, 3) and "lead" in o
print("G-load OK:", core.tag, "arm", core.model.arm, "poses at 4 s", o["poses"][-1].round(3).tolist())
' > "$R/$t.out" 2>&1 || die "G-load of $t failed: $(tail -3 "$R/$t.out" | tr '\n' ' ')"
    grep -q "G-load OK" "$R/$t.out" || die "G-load of $t: no OK line in $R/$t.out"
    date > "$R/DONE"; st "done: $(grep 'G-load OK' "$R/$t.out")";;
  *) die "unknown stage $stage";;
esac
