#!/usr/bin/env bash
# Lane ONPOL-NUPLAN trainings (plans/2026-10-11-onpol-prereg.md, "Arms" and "Pilot"), every job through the GPU pool, experiments/onpol_nuplan/scripts/on_train.py.
#   pilot  the 3 000-step pilots (seed 0, all shards, from P2H10S-F-s0), fanned out at once: SIMIT-ON-P-s0, SSELF-ON-P-s0, SBOTH-ON-P-s0, SNONE-ON-P-s0;
#          then on_pilot_check.py -> experiments/onpol_nuplan/results/pilot/pilot_gates.{json,md} (training-side gates; the 48-scene smoke is the lane's step)
#   full   SSELF-ON-s0 first; when its log reaches step 300 its first lines are checked (finite, 102 / 13 / 13, 13 hinge rows, no imitation row) and
#          written to STATUS, then the other seven (10 000 steps, seed s from P2H10S-F-s<s>): S{IMIT,SELF,BOTH,NONE}-ON-s{0,1}
#   scripts/tmux_run.sh on-pilot bash experiments/onpol_nuplan/scripts/on_train.sh pilot POOL
#   scripts/tmux_run.sh on-full  bash experiments/onpol_nuplan/scripts/on_train.sh full POOL
# POOL = the on-policy pool's cache dir name (plans/on_pool_format.md). Environment:
#   ON_LON=1    lane L is included (prereg "L inclusion rule"): SSELF / SBOTH get --on-lon, and the arm SSELFnoL-ON (self without it) replaces SNONE-ON
#               in `full` and is added as a fifth pilot
#   ARMS="SIMIT SSELF"   full: train only these families (default: all four); an arm whose pilot gate failed is skipped and named unless FORCE=1
#   VRAM=40     GB booked per training
# State: $DATA_DIR/runs/onpol_nuplan/train/<stage>/{STATUS, DONE | ERROR, log.txt, pool/<job>/}. Rerunning resumes: a finished job is not resubmitted, a
# queued or running one is not doubled, and on_train.py never overwrites an existing checkpoint.
set -uo pipefail
cd "$(dirname "$0")/../../.."
STAGE=${1:?pilot | full}; POOL=${2:?pool cache dir name}
D=$DATA_DIR/runs/onpol_nuplan/train/$STAGE; L=$D/pool; mkdir -p "$L"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
CL="$PWD/.venv/bin/python -m jevdrive.cl"
R=experiments/onpol_nuplan/results/pilot
CK=$DATA_DIR/runs/op_parity/runs
status() { echo "$(date '+%F %T') onpol train $STAGE: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1; shift; [[ -f $L/$n/DONE || -f $CK/$n/ckpt-final.pt ]] && return
        $CL queue 2>/dev/null | awk -v n="on-$n" '$4 == n && ($2 == "queued" || $2 == "running") {f = 1} END {exit !f}' && return; rm -f "$L/$n/ERROR"
        $CL submit --owner onpol-nuplan --name "on-$n" --train --priority 15 --vram "${VRAM:-40}" --cpu 6 --ram 60 --timeout-h 4 --log-dir "$L/$n" -- "$@" > /dev/null || die "submit $n"; }
waitdirs() { for n in "$@"; do until [[ -f $L/$n/DONE || -f $L/$n/ERROR ]]; do [[ -f $CK/$n/ckpt-final.pt && ! -d $L/$n ]] && break; sleep 30; done
             [[ -f $L/$n/ERROR ]] && die "job failed: $L/$n"; done; return 0; }
[[ -f $DATA_DIR/runs/op_parity/cache/$POOL/tab.npz && -f $DATA_DIR/runs/op_parity/cache/$POOL@warp/front.npy ]] || die "no pool $POOL"
S="--ho ot1:4,yr1:4,bd4:5 --ho-w 3 --ho-excl navsim/body1-val-logs --agent-lam 10 --shape --label-bank"
OT="$PY experiments/onpol_nuplan/scripts/on_train.py train $S"
ON="--on $POOL --on-k 13"
[[ $POOL == onstand_* ]] || ON="$ON --on-split alpasim_nuplan/onpol-train"
LON=""; [[ ${ON_LON:-0} == 1 ]] && LON="--on-lon"
declare -A F=([SIMIT]="$ON --on-mode imit" [SSELF]="$ON --on-mode self $LON" [SBOTH]="$ON --on-mode both $LON" [SNONE]="" [SSELFnoL]="$ON --on-mode self")
run() { local fam=$1 tag=$2 seed=$3; shift 3; sub "$tag" $OT ${F[$fam]} --seed "$seed" --init "P2H10S-F-s$seed" --tag "$tag" "$@"; }
status "code $(git rev-parse --short HEAD); pool $POOL; lon ${ON_LON:-0}"
if [[ $STAGE == pilot ]]; then
  P=(SIMIT SSELF SBOTH SNONE); [[ -n $LON ]] && P+=(SSELFnoL)
  for f in "${P[@]}"; do run "$f" "$f-ON-P-s0" 0 --steps 3000 --eval-every 1000; done
  status "${#P[@]} pilots submitted (3 000 steps); waiting"
  waitdirs "${P[@]/%/-ON-P-s0}"
  T=("${P[@]/%/-ON-P-s0}"); T=("${T[@]/SNONE-ON-P-s0}")
  $PY experiments/onpol_nuplan/scripts/on_pilot_check.py --tags ${T[*]} --none SNONE-ON-P-s0 --out "$R" > "$D/pilot_gates.txt"; rc=$?
  cat "$D/pilot_gates.txt"
  [[ $rc -eq 0 ]] || die "pilot gates: exit $rc (2 = an arm fails, 3 = no ON-NONE pilot); see $R/pilot_gates.md"
  status "pilots pass the training-side gates: $R/pilot_gates.md"
elif [[ $STAGE == full ]]; then
  A=(${ARMS:-SIMIT SSELF SBOTH $([[ -n $LON ]] && echo SSELFnoL || echo SNONE)})
  ok=()
  for f in "${A[@]}"; do
    if [[ ${FORCE:-0} == 1 ]] || python3 -c "import json,sys; sys.exit(not json.load(open('$R/pilot_gates.json'))['arms']['$f-ON-P-s0']['pass'])" 2>/dev/null; then ok+=("$f")
    else status "skipped $f: its pilot did not pass (or $R/pilot_gates.json is missing); FORCE=1 overrides"; fi
  done
  [[ " ${ok[*]} " == *" SSELF "* ]] || die "SSELF is not trainable: the first full run is SSELF-ON-s0"
  run SSELF SSELF-ON-s0 0
  status "SSELF-ON-s0 submitted; waiting for its step 300"
  until grep -q "step 300:" "$L/SSELF-ON-s0/log.txt" 2>/dev/null || [[ -f $L/SSELF-ON-s0/DONE || -f $L/SSELF-ON-s0/ERROR || ( -f $CK/SSELF-ON-s0/ckpt-final.pt && ! -d $L/SSELF-ON-s0 ) ]]; do sleep 30; done
  [[ -f $L/SSELF-ON-s0/ERROR ]] && die "job failed: $L/SSELF-ON-s0"
  if [[ -f $L/SSELF-ON-s0/log.txt ]]; then
    grep -h "per batch of\|dev @ 0\|step [123]00:" "$L/SSELF-ON-s0/log.txt" | sed 's/.*INFO *//' | cut -c1-700 | tee "$D/first300.txt"
    grep -q "per batch of 128: 102 normal, 13 hinge-only .* 13 on-policy (mode self" "$D/first300.txt" || die "SSELF-ON-s0: composition"
    [[ $(grep -c "n_hinge 13.0000, n_imit 0.0000" "$D/first300.txt") -ge 3 ]] || die "SSELF-ON-s0: on-policy row accounting in the first 300 steps"
    grep "step [123]00:" "$D/first300.txt" | grep -qi "nan\|inf " && die "SSELF-ON-s0: non-finite loss"
  fi
  status "SSELF-ON-s0 first 300 steps ok ($D/first300.txt); the rest: ${ok[*]} x seeds 0 1"
  tags=()
  for f in "${ok[@]}"; do for s in 0 1; do run "$f" "$f-ON-s$s" "$s"; tags+=("$f-ON-s$s"); done; done
  waitdirs "${tags[@]}"
  for t in "${tags[@]}"; do [[ -f $CK/$t/ckpt-final.pt ]] || die "no checkpoint for $t"; done
  grep -h "dev @ 10000" "$L"/*/log.txt | sed 's/.*INFO *//' | cut -c1-200
  status "${#tags[@]} checkpoints: ${tags[*]}"
else
  die "stage $STAGE: pilot | full"
fi
$CL usage --hours 12 > "$D/usage.txt" 2>&1 || true
date '+%F %T' > "$D/DONE"
