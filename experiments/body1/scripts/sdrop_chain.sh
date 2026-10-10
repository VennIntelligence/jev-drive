#!/usr/bin/env bash
# S-DROP (experiments/body1/plans/2026-10-10-sdrop-prereg.md): full-scale drop-one ablation of P2H10S, three arms x two seeds, read on navhard (G),
# navtest, the turn-oracle replay and the own-plan readers. One self-advancing chain: it only submits pool jobs and waits on their DONE / ERROR.
#   scripts/tmux_run.sh sdrop bash experiments/body1/scripts/sdrop_chain.sh
# State: $DATA_DIR/runs/body1/sdrop/chain/{STATUS, DONE | ERROR, log.txt, jobs.txt}; job logs under chain/pool/<job>/. Rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/body1/sdrop
D=$O/chain; L=$D/pool; mkdir -p "$L"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
NAV=$DATA_DIR/envs/navsim2/bin/python
VPY=$PWD/.venv/bin/python
CL="$DATA_DIR/envs/jevdrive/bin/python -m jevdrive.cl"
S=experiments/body1/scripts
SP=experiments/op_parity/scripts
BN=("$VPY" -m jevdrive.bench)
R=experiments/body1/results/sdrop; mkdir -p "$R/bench"
ARMS=(noA noB noC)
HO="--ho ot1:4,yr1:4,bd4:5 --ho-w 3 --ho-excl navsim/body1-val-logs"
declare -A F=([S]="$HO --agent-lam 10 --shape" [noA]="$HO --shape" [noB]="--agent-lam 10 --shape" [noC]="$HO --agent-lam 10 --road-lam 0 --shape")
ALL=$(for k in $(seq 0 11); do echo -n "navtrain_full.s${k}of12 "; done)
status() { echo "$(date '+%F %T') body1 sdrop: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && return; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner body1-sdrop --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 20; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; return 0; }
run_dir() { ls -d "$DATA_DIR/runs/op_parity/train-$1"/* | tail -1; }

status "stage 1: identity run and one smoke per arm (60 steps, shard s2)"
T60="--seed 0 --data navtrain_full.s2of12 --steps 60 --eval-every 30"
sub sdrop-ident "$L/ident" --train --vram 20 --cpu 4 --ram 30 --timeout-h 1 -- $PY $S/bd4_train.py train $T60 ${F[S]} --tag SDROP-IDENT-S
for a in "${ARMS[@]}"; do
  sub sdrop-smoke-$a "$L/smoke-$a" --train --vram 20 --cpu 4 --ram 30 --timeout-h 1 -- $PY $S/bd4_train.py train $T60 ${F[$a]} --tag SDROP-SMOKE-$a
done
waitdirs "$L/ident" "$L/smoke-noA" "$L/smoke-noB" "$L/smoke-noC"
if ! $PY $S/bd4_train.py ident --a "$(run_dir B43A7-IDENT-S1)" --b "$(run_dir SDROP-IDENT-S)" --out $R/ident_sdrop.json; then
  # prereg 3.1: a second run of the same command decides between a code difference and run-to-run nondeterminism
  sub sdrop-ident2 "$L/ident2" --train --vram 20 --cpu 4 --ram 30 --timeout-h 1 -- $PY $S/bd4_train.py train $T60 ${F[S]} --tag SDROP-IDENT-S2
  waitdirs "$L/ident2"
  $PY $S/bd4_train.py ident --a "$(run_dir SDROP-IDENT-S)" --b "$(run_dir SDROP-IDENT-S2)" --out $R/ident_sdrop_repeat.json \
    && die "trainer with all three terms is not bit-identical to the stored P2H10S code path (the repeat is identical to itself)"
  status "identity: the box is not run-to-run deterministic (ident_sdrop_repeat.json); config identity only"
fi
$PY $S/sdrop_report.py smoke || die "smoke checklist"

status "stage 2: six full runs"
for a in "${ARMS[@]}"; do for s in 0 1; do
  sub sdrop-full-$a-s$s "$L/full-$a-s$s" --train --vram 32 --cpu 4 --ram 56 --timeout-h 3 -- $PY $S/bd4_train.py train --seed $s --data $ALL --steps 10000 ${F[$a]} --tag P2H10S-$a-F-s$s
done; done
waitdirs $(for a in "${ARMS[@]}"; do for s in 0 1; do echo "$L/full-$a-s$s"; done; done)
$PY $S/sdrop_report.py trained || die "config identity / training sanity"

status "stage 3: reads (bench navtest + navhard G, own-plan readers)"
NEW=$(for a in "${ARMS[@]}"; do echo -n "P2H10S-$a-F-s0 P2H10S-$a-F-s1 "; done)
"${BN[@]}" run --model $NEW --bench navtest || die "bench navtest"
"${BN[@]}" run --model $(for t in $NEW; do echo -n "$t@gimm "; done) --bench navhard || die "bench navhard"
for a in "${ARMS[@]}"; do for s in 0 1; do
  sub sdrop-g3h-$a-s$s "$L/g3-hold-$a-s$s" --vram 10 --cpu 2 --ram 20 -- $PY $S/bd4_g3.py --name sdrop_hold_${a}_s$s --set hold --new P2H10S-$a-F-s$s --ref P2H10-F-s$s P2H10S-F-s$s --out $R
  sub sdrop-g3n-$a-s$s "$L/g3-navtest-$a-s$s" --vram 10 --cpu 2 --ram 12 -- $PY $S/bd4_g3.py --name sdrop_navtest_${a}_s$s --set navtest --new P2H10S-$a-F-s$s --ref P2H10-F-s$s P2H10S-F-s$s --out $R
done; done
NEW8="$NEW P2H10S-F-s0 P2H10S-F-s1"
REF8="P2H10-F-s0 P2H10-F-s1 P2H10-F-s0 P2H10-F-s1 P2H10-F-s0 P2H10-F-s1 P2H10-F-s0 P2H10-F-s1"
for st in hold navtest; do
  sub sdrop-dump-$st "$L/dump-$st" --vram 20 --cpu 2 --ram 28 -- $PY $S/prog_ol.py states --set $st --new $NEW8 --ref $REF8 --out $R --name sdrop_$st
done
waitdirs "$L/dump-hold" "$L/dump-navtest" $(for a in "${ARMS[@]}"; do for s in 0 1; do echo "$L/g3-hold-$a-s$s $L/g3-navtest-$a-s$s"; done; done)
for st in hold navtest; do
  sub sdrop-route-$st "$L/route-$st" --vram 0.5 --cpu 8 --ram 16 -- $PY $S/route_ol.py --name sdrop_$st --set $st --new $NEW8 --ref $REF8 --out $R
done
"${BN[@]}" status --model $NEW --bench navtest --wait || die "navtest"
sub sdrop-replay "$L/replay" --vram 0.5 --cpu 48 --ram 64 -- $NAV $SP/turn_oracle.py replay --name sdrop --models $NEW P2H10S-F-s0 P2H10S-F-s1 P2H10-F-s0 P2H10-F-s1
"${BN[@]}" status --model $(for t in $NEW; do echo -n "$t@gimm "; done) --bench navhard --wait || die "navhard"
waitdirs "$L/replay" "$L/route-hold" "$L/route-navtest"

status "stage 4: tables"
arm() { echo "$1=P2H10${2}-F-s0$3+P2H10${2}-F-s1$3"; }
"${BN[@]}" report --bench navtest --arms $(arm noA S-noA) $(arm noB S-noB) $(arm noC S-noC) --vs $(arm S S) $(arm base "") --out $R/bench || die "report navtest"
"${BN[@]}" report --bench navhard --arms $(arm noA S-noA @gimm) $(arm noB S-noB @gimm) $(arm noC S-noC @gimm) --vs $(arm S S @gimm) $(arm base "" @gimm) --out $R/bench || die "report navhard"
$PY $S/bd4_g3d.py --replay sdrop --new P2H10S-F --base P2H10-F --out $R/d_S_vs_base > "$D/g3d.txt" || die "g3d S"
for a in "${ARMS[@]}"; do
  $PY $S/bd4_g3d.py --replay sdrop --new P2H10S-$a-F --base P2H10-F --out $R/d_${a}_vs_base >> "$D/g3d.txt" || die "g3d $a base"
  $PY $S/bd4_g3d.py --replay sdrop --new P2H10S-$a-F --base P2H10S-F --out $R/d_${a}_vs_S >> "$D/g3d.txt" || die "g3d $a S"
done
sub sdrop-report "$L/report" --vram 0.5 --cpu 8 --ram 24 -- $PY $S/sdrop_report.py report
waitdirs "$L/report"
$CL usage --hours 6 > "$D/usage.txt" 2>&1 || true
status "done"; date '+%F %T' > "$D/DONE"
