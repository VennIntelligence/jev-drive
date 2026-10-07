#!/usr/bin/env bash
# op_parity / wod-pref lane (plans/2026-10-08-wod-pref-prereg.md): rater preference fine-tuned into System 1 (WLG) on WOD val by 5-fold.
# One self-advancing chain in tmux jev (scripts/tmux_run.sh); every GPU job goes through the pool. Rerunning resumes (finished folds are skipped).
#   prep    rater-frame token cache + gates P0 / P1
#   pilot   fold 0, objective top, seed 0, three learning rates -> the learning rate (inner fold only)
#   full    4 objectives x 2 seeds x 5 folds + the permutation control (seed 0) -> report, figures
#   final   every objective whose out-of-fold CI excludes 0: all 479 frames fitted, 2 seeds, checkpoints WPF-<obj>-all-s<seed>
#   all     prep -> pilot -> full -> final
# State: $DATA_DIR/runs/op_parity/wod/pref/{STATUS, DONE-<stage>, ERROR, chain.log, jobs.txt}.
set -uo pipefail
cd "$(dirname "$0")/../../.."
STAGE=${1:?stage: prep | pilot | full | final | all}
D=$DATA_DIR/runs/op_parity/wod/pref; mkdir -p "$D"; rm -f "$D/DONE-$STAGE" "$D/ERROR"
exec > >(tee -a "$D/chain.log") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
J=$DATA_DIR/envs/jevdrive/bin/python
CL="$J -m jevdrive.cl"
S=experiments/op_parity/scripts
L=$D/pool
OBJS="top rank hinge f20"
status() { echo "$(date '+%F %T') op_parity wod_pref: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && { echo done; return; }
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && { echo "$live"; return; }; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; echo "$id"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 20; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; }
train() { local name=$1; shift; sub wpf-$name $L/$name --train --vram 18 --cpu 3 --ram 16 -- $PY $S/wod_pref.py train "$@" >/dev/null; }

prep() {
  status "prep: rater token cache"
  sub wpf-prep $L/prep --vram 10 --cpu 6 --ram 24 -- $PY $S/wod_pref.py prep --workers 4 >/dev/null
  waitdirs $L/prep
  status "prep: done ($(tr -d '\n ' < $DATA_DIR/runs/op_parity/cache/wod_rater/gates.json))"
}
pilot() {
  status "pilot: fold 0, top, seed 0, three learning rates"
  local dirs=""
  for lr in "3e-06 3e-05" "1e-05 1e-04" "3e-05 3e-04"; do set -- $lr
    train pilot-lr$1 --obj top --seed 0 --folds 0 --lr $1 --lr-new $2 --tag pilot-lr$1; dirs="$dirs $L/pilot-lr$1"; done
  waitdirs $dirs
  $PY $S/wod_pref.py pick || die "pick"
  status "pilot: done ($(cat experiments/op_parity/results/wod_pref/pilot.json))"
}
full() {
  local lr lrn dirs=""
  read -r lr lrn < <($J -c "import json; d = json.load(open('experiments/op_parity/results/wod_pref/pilot.json')); print(d['lr'], d['lr_new'])") || die "no pilot.json"
  status "full: lr $lr / $lrn, 4 objectives x 2 seeds + permutation control"
  for o in $OBJS; do
    for s in 0 1; do train $o-s$s --obj $o --seed $s --lr $lr --lr-new $lrn; dirs="$dirs $L/$o-s$s"; done
    train $o-s0-perm --obj $o --seed 0 --perm --lr $lr --lr-new $lrn; dirs="$dirs $L/$o-s0-perm"
  done
  waitdirs $dirs
  status "full: report"
  $J $S/wod_pref.py report || die "report"
  $J $S/wod_pref.py figs || die "figs"
  status "full: done"
}
final() {
  local lr lrn dirs="" n=0
  read -r lr lrn < <($J -c "import json; d = json.load(open('experiments/op_parity/results/wod_pref/pilot.json')); print(d['lr'], d['lr_new'])") || die "no pilot.json"
  while read -r o st; do
    [[ -z $o ]] && continue
    for s in 0 1; do train WPF-$o-all-s$s --obj $o --seed $s --folds all --steps $st --lr $lr --lr-new $lrn --tag WPF-$o-all-s$s </dev/null; dirs="$dirs $L/WPF-$o-all-s$s"; done
    n=$((n + 1))
  done < <($J $S/wod_pref.py final-plan)
  [[ $n == 0 ]] && { status "final: no objective with an out-of-fold CI above 0, no checkpoint trained"; return; }
  waitdirs $dirs
  status "final: $n objective(s) trained on all 479 frames"
}
case $STAGE in
  prep) prep ;; pilot) pilot ;; full) full ;; final) final ;;
  all) prep; pilot; full; final ;;
  *) die "unknown stage $STAGE" ;;
esac
touch "$D/DONE-$STAGE"; status "$STAGE: DONE"
