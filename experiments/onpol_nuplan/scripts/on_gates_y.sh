#!/usr/bin/env bash
# Lane ONPOL-NUPLAN round 2 (plans/2026-10-12-onpol2-prereg.md, B "Trainer gates"): on_train.py with --yr-mass (the YR10m10-F base), pool jobs.
#   Y1  60 steps, seed 0, one shard, no --init, no --on, --yr-mass 0.1, no S flag: bit-identical to `joint_train.py --yr-mass 0.1` (ident), with
#       and without --label-bank (the arms use the bank)
#   Y2  --init YR10m10-F-s0, all shards: every tensor loaded, dev ADE at step 0 = the archived run's dev @ 10000 (0.539) to 3 decimals
#   Y3  two `self` runs of the YSELFnoL recipe on POOL (300 and 60 steps): on-policy rows carry both hinges and no imitation term, no offline row
#       enters the agent hinge, the same on-policy indices (running sha1 at step 60)
#   R   round-1 regression: gate T1 again (S flags, no on-policy rows, against joint_train.py), both runs fresh
#   scripts/tmux_run.sh on-gates-y bash experiments/onpol_nuplan/scripts/on_gates_y.sh [POOL [GEN]]      (default on1_train, GEN "")
# State: $DATA_DIR/runs/onpol_nuplan/gates-y<GEN>/{STATUS, DONE | ERROR, log.txt, pool/<job>/}; results experiments/onpol_nuplan/results/gates/y_*.json.
# Rerunning resumes; tags ONY<GEN>-* are never overwritten (a new code version takes a new GEN).
set -uo pipefail
cd "$(dirname "$0")/../../.."
POOL=${1:-on1_train}; G=ONY${2:-}
D=$DATA_DIR/runs/onpol_nuplan/gates-y${2:-}; L=$D/pool; mkdir -p "$L"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
CL="$PWD/.venv/bin/python -m jevdrive.cl"
R=experiments/onpol_nuplan/results/gates; mkdir -p "$R"
status() { echo "$(date '+%F %T') onpol gates-y: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 v=$2; shift 2; [[ -f $L/$n/DONE ]] && return
        $CL queue 2>/dev/null | awk -v n="ony-$n" '$4 == n && ($2 == "queued" || $2 == "running") {f = 1} END {exit !f}' && return; rm -f "$L/$n/ERROR"
        $CL submit --owner onpol-nuplan-trainer --name "ony-$n" --train --priority 15 --vram "$v" --cpu 6 --ram 60 --timeout-h 1 --log-dir "$L/$n" -- "$@" > /dev/null || die "submit $n"; }
waitdirs() { for n in "$@"; do until [[ -f $L/$n/DONE || -f $L/$n/ERROR ]]; do sleep 20; done; [[ -f $L/$n/ERROR ]] && die "job failed: $L/$n"; done; return 0; }
status "code $(git rev-parse --short HEAD); pool $POOL"
[[ -f $DATA_DIR/runs/op_parity/cache/$POOL/road.npz ]] || die "no pool $POOL"
S="--ho ot1:4,yr1:4,bd4:5 --ho-w 3 --ho-excl navsim/body1-val-logs --agent-lam 10 --shape --label-bank"
Y="--yr-mass 0.1 --label-bank"
YS="$Y --agent-lam 10 --agent-on-only --shape --ho-w 3 --on $POOL --on-k 13 --on-mode self"
SH="--data navtrain_full.s2of12 --split navsim/op-parity-full --batch 128 --warmup 300 --eval-every 30 --steps 60 --seed 0"
JT="$PY experiments/alpasim/scripts/joint_train.py train --exp onpol_nuplan"
OT="$PY experiments/onpol_nuplan/scripts/on_train.py"
sub y1-jt 20 $JT $SH --yr-mass 0.1 --tag $G-Y1-jt
sub y1-on 20 $OT train $SH --yr-mass 0.1 --tag $G-Y1-on
sub y1-onb 20 $OT train $SH $Y --tag $G-Y1-onb
sub t1-jt 20 $JT $SH $S --tag $G-T1-jt
sub t1-on 20 $OT train $SH $S --tag $G-T1-on
FL="--seed 0 --init YR10m10-F-s0"
sub f-none 40 $OT train $FL $Y --steps 300 --eval-every 300 --tag $G-none
sub f-self 40 $OT train $FL $YS --steps 300 --eval-every 300 --tag $G-self
sub f-selfb 40 $OT train $FL $YS --steps 60 --eval-every 60 --tag $G-selfb
status "8 jobs submitted; waiting"
waitdirs y1-jt y1-on y1-onb t1-jt t1-on f-none f-self f-selfb
fail=()
$OT ident --a $G-Y1-jt --b $G-Y1-on --out "$R/y_y1_ident.json" > /dev/null || fail+=(Y1)
$OT ident --a $G-Y1-jt --b $G-Y1-onb --out "$R/y_y1_ident_bank.json" > /dev/null || fail+=(Y1-bank)
$OT ident --a $G-T1-jt --b $G-T1-on --out "$R/y_t1_round1_ident.json" > /dev/null || fail+=(T1-round1)
$OT t2 --tags $G-none $G-self --ref "${REF:-0.539}" --out "$R/y_y2_init_perf.json" > /dev/null || fail+=(Y2)
$OT y3 --tags $G-self $G-selfb --step 60 --out "$R/y_y3_rows.json" > /dev/null || fail+=(Y3)
grep -h "per batch of\|dev @\|on_hash @ 60\|step 300:\|running on card" "$L"/f-*/log.txt "$L"/f-*/STATUS 2>/dev/null | sed 's/.*INFO *//' | cut -c1-600
[[ ${#fail[@]} -eq 0 ]] || die "gates failed: ${fail[*]} (see $R/y_*.json)"
status "Y1, Y2, Y3 and the round-1 T1 pass"; date '+%F %T' > "$D/DONE"
