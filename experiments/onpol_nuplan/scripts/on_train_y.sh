#!/usr/bin/env bash
# Lane ONPOL-NUPLAN round 2 trainings (plans/2026-10-12-onpol2-prereg.md, B and C), every job through the GPU pool, scripts/on_train.py.
# The Y family continues the archived YR10m10-F-s<seed> with its own recipe (--yr-mass 0.1: 13 yaw-rate rows per batch, no lesson S):
#   YSELFnoL-ON  102 normal / 13 yaw-rate / 13 on-policy rows; the on-policy rows (pool POOL, stream [seed, 0, 53]) are hinge-only and carry S's two
#                hinges (agent against the pool's actor boxes, road-and-lane raster; lambda 10 each, shape-only, x 3); no offline row carries an
#                agent hinge (--agent-on-only), no hinge-only offline row
#   YNONE-ON     115 normal / 13 yaw-rate, no on-policy rows: the same continuation (`joint_train.py --yr-mass 0.1` from the checkpoint)
# Stages:
#   pilot [POOL]  YSELFnoL-ON-P-s0 + YNONE-ON-P-s0 (3 000 steps, all shards, from YR10m10-F-s0), then on_pilot_check.py with the YNONE pilot as the
#                 reference -> experiments/onpol_nuplan/results/pilot_y/pilot_gates.{json,md}. Exit 1 if a gate fails: item B stops there.
#                 The 48-scene mechanics smoke is on_read.sh smoke YSELFnoL-ON-P-s0 YNONE-ON-P-s0 (the lane's step).
#   full [POOL]   YSELFnoL-ON-s{0,1}, YNONE-ON-s{0,1} (10 000 steps, seed s from YR10m10-F-s<s>). Starts only if every arm of pilot_gates.json passes
#                 and both smoke checklists ($DATA_DIR/runs/onpol_nuplan/read/smoke/smoke_<pilot tag>.json) pass; there is no override (the prereg:
#                 a failed gate stops item B). YSELFnoL-ON-s0 goes first and its first 300 steps are checked before the other three.
#   c             prereg section C: SSELFnoL-ON-s2 / -s3 from P2H10S-F-s2 / -s3, round 1's recipe and pool on1_train unchanged (on_train.sh's SSELFnoL
#                 line). Not gated on B; the lane starts it when section C's condition holds.
#   scripts/tmux_run.sh on-ypilot bash experiments/onpol_nuplan/scripts/on_train_y.sh pilot
#   scripts/tmux_run.sh on-yfull  bash experiments/onpol_nuplan/scripts/on_train_y.sh full
#   scripts/tmux_run.sh on-c      bash experiments/onpol_nuplan/scripts/on_train_y.sh c
# POOL defaults to on1_trainY. Environment: VRAM=40 (GB booked per training); SFX=-smoke PILOT_STEPS=100 FULL_STEPS=400 = chain smoke (tags <tag><SFX>,
# state dir <stage><SFX>, the gate table under the state dir, and `full` then needs no closed-loop smoke checklist).
# State: $DATA_DIR/runs/onpol_nuplan/train/y<stage>/{STATUS, DONE | ERROR, log.txt, pool/<job>/}. Rerunning resumes: a finished job is not resubmitted,
# a queued or running one is not doubled, and on_train.py never overwrites an existing checkpoint.
set -uo pipefail
cd "$(dirname "$0")/../../.."
STAGE=${1:?pilot | full | c}; POOL=${2:-on1_trainY}
X=${SFX:-}; NP=${PILOT_STEPS:-3000}; NF=${FULL_STEPS:-10000}
D=$DATA_DIR/runs/onpol_nuplan/train/y$STAGE$X; L=$D/pool; mkdir -p "$L"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
CL="$PWD/.venv/bin/python -m jevdrive.cl"
R=experiments/onpol_nuplan/results/pilot_y; [[ -n $X ]] && R=$DATA_DIR/runs/onpol_nuplan/train/ypilot$X/results
CK=$DATA_DIR/runs/op_parity/runs
status() { echo "$(date '+%F %T') onpol train y$STAGE: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1; shift; [[ -f $L/$n/DONE || -f $CK/$n/ckpt-final.pt ]] && return
        $CL queue 2>/dev/null | awk -v n="on-$n" '$4 == n && ($2 == "queued" || $2 == "running") {f = 1} END {exit !f}' && return; rm -f "$L/$n/ERROR"
        $CL submit --owner onpol-nuplan --name "on-$n" --train --priority 15 --vram "${VRAM:-40}" --cpu 6 --ram 60 --timeout-h 4 --log-dir "$L/$n" -- "$@" > /dev/null || die "submit $n"; }
waitdirs() { for n in "$@"; do until [[ -f $L/$n/DONE || -f $L/$n/ERROR ]]; do [[ -f $CK/$n/ckpt-final.pt && ! -d $L/$n ]] && break; sleep 30; done
             [[ -f $L/$n/ERROR ]] && die "job failed: $L/$n"; done; return 0; }
OT="$PY experiments/onpol_nuplan/scripts/on_train.py train"
[[ $STAGE == c ]] && POOL=on1_train
[[ -f $DATA_DIR/runs/op_parity/cache/$POOL/tab.npz && -f $DATA_DIR/runs/op_parity/cache/$POOL@warp/front.npy ]] || die "no pool $POOL"
ON="--on $POOL --on-k 13 --on-split alpasim_nuplan/onpol-train --on-mode self"
Y="--yr-mass 0.1 --label-bank"
declare -A F=([YSELFnoL]="$Y --agent-lam 10 --agent-on-only --shape --ho-w 3 $ON" [YNONE]="$Y")
run() { local fam=$1 tag=$2 seed=$3; shift 3; sub "$tag" $OT ${F[$fam]} --seed "$seed" --init "YR10m10-F-s$seed" --tag "$tag" "$@"; }
passed() { python3 -c "import json,sys; sys.exit(not json.load(open(sys.argv[1]))$2)" "$1" 2>/dev/null; }
status "code $(git rev-parse --short HEAD); pool $POOL"
if [[ $STAGE == pilot ]]; then
  P=(YSELFnoL YNONE)
  for f in "${P[@]}"; do run "$f" "$f-ON-P-s0$X" 0 --steps "$NP" --eval-every 1000; done
  status "2 pilots submitted ($NP steps); waiting"
  waitdirs "${P[@]/%/-ON-P-s0$X}"
  $PY experiments/onpol_nuplan/scripts/on_pilot_check.py --tags "YSELFnoL-ON-P-s0$X" --none "YNONE-ON-P-s0$X" --steps "$NP" --ho-k 0 --yr-k 13 \
      --archived 0.539 --archived-name YR10m10-F-s0 --out "$R" > "$D/pilot_gates.txt"; rc=$?
  cat "$D/pilot_gates.txt"
  [[ $rc -eq 0 ]] || die "pilot gates: exit $rc (2 = an arm fails, 3 = no YNONE pilot); item B stops (prereg); see $R/pilot_gates.md"
  status "pilots pass the training-side gates: $R/pilot_gates.md; next: on_read.sh smoke YSELFnoL-ON-P-s0$X YNONE-ON-P-s0$X"
elif [[ $STAGE == full ]]; then
  for t in YSELFnoL-ON-P-s0$X YNONE-ON-P-s0$X; do
    passed "$R/pilot_gates.json" "['arms']['$t']['pass']" || die "pilot gate of $t not passed (or $R/pilot_gates.json missing): item B stops, no override"
    [[ -n $X ]] || passed "$DATA_DIR/runs/onpol_nuplan/read/smoke/smoke_$t.json" "['pass']" || die "48-scene smoke checklist of $t not passed (on_read.sh smoke): no override"
  done
  T0=YSELFnoL-ON-s0$X
  run YSELFnoL $T0 0 --steps "$NF"
  status "$T0 submitted; waiting for its step 300"
  until grep -q "step 300:" "$L/$T0/log.txt" 2>/dev/null || [[ -f $L/$T0/DONE || -f $L/$T0/ERROR || ( -f $CK/$T0/ckpt-final.pt && ! -d $L/$T0 ) ]]; do sleep 30; done
  [[ -f $L/$T0/ERROR ]] && die "job failed: $L/$T0"
  if [[ -f $L/$T0/log.txt ]]; then
    grep -h "per batch of\|dev @ 0\|step [123]00:" "$L/$T0/log.txt" | sed 's/.*INFO *//' | cut -c1-900 | tee "$D/first300.txt"
    grep -q "per batch of 128: 102 normal, 13 yaw-rate, 0 hinge-only .* 13 on-policy (mode self" "$D/first300.txt" || die "$T0: composition"
    [[ $(grep -c "n_hinge 13.0000, n_imit 0.0000, off_agent_rows 0.0000" "$D/first300.txt") -ge 3 ]] || die "$T0: on-policy row accounting in the first 300 steps"
    grep "step [123]00:" "$D/first300.txt" | grep -qi "nan\|inf " && die "$T0: non-finite loss"
  fi
  status "$T0 first 300 steps ok ($D/first300.txt); the rest"
  tags=()
  for f in YSELFnoL YNONE; do for s in 0 1; do run "$f" "$f-ON-s$s$X" "$s" --steps "$NF"; tags+=("$f-ON-s$s$X"); done; done
  waitdirs "${tags[@]}"
  for t in "${tags[@]}"; do [[ -f $CK/$t/ckpt-final.pt ]] || die "no checkpoint for $t"; done
  grep -h "dev @ $NF" "$L"/*/log.txt | sed 's/.*INFO *//' | cut -c1-200
  status "${#tags[@]} checkpoints: ${tags[*]}"
elif [[ $STAGE == c ]]; then
  S="--ho ot1:4,yr1:4,bd4:5 --ho-w 3 --ho-excl navsim/body1-val-logs --agent-lam 10 --shape --label-bank"       # on_train.sh, arm SSELFnoL
  tags=()
  for s in 2 3; do t=SSELFnoL-ON-s$s$X; sub "$t" $OT $S --on on1_train --on-k 13 --on-split alpasim_nuplan/onpol-train --on-mode self --seed $s --init "P2H10S-F-s$s" --tag "$t" --steps "$NF"; tags+=("$t"); done
  status "${tags[*]} submitted ($NF steps); waiting"
  waitdirs "${tags[@]}"
  for t in "${tags[@]}"; do [[ -f $CK/$t/ckpt-final.pt ]] || die "no checkpoint for $t"; done
  grep -h "per batch of\|dev @ 0\|dev @ $NF" "$L"/*/log.txt | sed 's/.*INFO *//' | cut -c1-400
  status "${#tags[@]} checkpoints: ${tags[*]}"
else
  die "stage $STAGE: pilot | full | c"
fi
$CL usage --hours 12 > "$D/usage.txt" 2>&1 || true
date '+%F %T' > "$D/DONE"
