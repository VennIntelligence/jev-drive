#!/usr/bin/env bash
# Lane ONPOL-NUPLAN: the trainer gates of experiments/onpol_nuplan/scripts/on_train.py (plans/2026-10-11-onpol-prereg.md, "Arms": T1 - T3) on the
# stand-in pool (on_train.py standin), plus steps / s and peak card memory per mode.
#   T1  60 steps, seed 0, one shard, no --init, no --on: bit-identical to joint_train.py with the S flags (joint_train.py ident)
#   T2  --init P2H10S-F-s0, all shards: every tensor loaded, dev ADE at step 0 = the archived 0.545 to 3 decimals
#   T3  imit / self / both draw the same on-policy indices over 60 steps (running sha1), per-mode row accounting, and the known-answer test
#   scripts/tmux_run.sh on-gates bash experiments/onpol_nuplan/scripts/on_gates.sh [POOL]      (default onstand_navtrain_full.s2of12)
# State: $DATA_DIR/runs/onpol_nuplan/gates/{STATUS, DONE | ERROR, log.txt, pool/<job>/}; results experiments/onpol_nuplan/results/gates/*.json.
# Rerunning resumes (finished jobs are not resubmitted; their tags exist, a tag is never overwritten).
set -uo pipefail
cd "$(dirname "$0")/../../.."
POOL=${1:-onstand_navtrain_full.s2of12}
D=$DATA_DIR/runs/onpol_nuplan/gates; L=$D/pool; mkdir -p "$L"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
CL="$PWD/.venv/bin/python -m jevdrive.cl"
R=experiments/onpol_nuplan/results/gates; mkdir -p "$R"
status() { echo "$(date '+%F %T') onpol gates: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 v=$2; shift 2; [[ -f $L/$n/DONE ]] && return
        $CL queue 2>/dev/null | awk -v n="ong-$n" '$4 == n && ($2 == "queued" || $2 == "running") {f = 1} END {exit !f}' && return; rm -f "$L/$n/ERROR"
        $CL submit --owner onpol-nuplan-trainer --name "ong-$n" --train --priority 15 --vram "$v" --cpu 6 --ram 60 --timeout-h 1 --log-dir "$L/$n" -- "$@" > /dev/null || die "submit $n"; }
waitdirs() { for n in "$@"; do until [[ -f $L/$n/DONE || -f $L/$n/ERROR ]]; do sleep 20; done; [[ -f $L/$n/ERROR ]] && die "job failed: $L/$n"; done; return 0; }
status "code $(git rev-parse --short HEAD); pool $POOL"
[[ -f $DATA_DIR/runs/op_parity/cache/$POOL/road.npz ]] || die "no pool $POOL (on_train.py standin)"
S="--ho ot1:4,yr1:4,bd4:5 --ho-w 3 --ho-excl navsim/body1-val-logs --agent-lam 10 --shape --label-bank"
SH="--data navtrain_full.s2of12 --split navsim/op-parity-full --batch 128 --warmup 300 --eval-every 30 --steps 60 --seed 0"
OT="$PY experiments/onpol_nuplan/scripts/on_train.py"
sub t1-jt 20 $PY experiments/alpasim/scripts/joint_train.py train --exp onpol_nuplan $SH $S --tag ONG-T1-jt
sub t1-on 20 $OT train $SH $S --tag ONG-T1-on
sub known 20 $OT known --data navtrain_full.s2of12 $S --init P2H10S-F-s0 --on "$POOL" --out "$R/t3_known.json"
FL="--steps 300 --eval-every 300 --seed 0 --init P2H10S-F-s0 $S"       # all shards: T2, T3's hash and row accounting, steps / s
for m in imit self both; do sub "f-$m" 40 $OT train $FL --on "$POOL" --on-mode $m --tag "ONG-$m"; done
sub f-none 40 $OT train $FL --tag ONG-none
status "7 jobs submitted; waiting"
waitdirs t1-jt t1-on known f-imit f-self f-both f-none
fail=()
$OT ident --a ONG-T1-jt --b ONG-T1-on --out "$R/t1_ident.json" > /dev/null || fail+=(T1)
$OT t2 --tags ONG-none ONG-imit ONG-self ONG-both --out "$R/t2_init_perf.json" > /dev/null || fail+=(T2)
$OT t3 --tags ONG-imit ONG-self ONG-both --step 60 --out "$R/t3_modes.json" > /dev/null || fail+=(T3)
grep -h "dev @\|step 60:\|step 300:" "$L"/f-*/log.txt | sed 's/.*INFO *//' | cut -c1-400
[[ ${#fail[@]} -eq 0 ]] || die "gates failed: ${fail[*]} (see $R/*.json)"
status "T1, T2, T3 pass"; date '+%F %T' > "$D/DONE"
