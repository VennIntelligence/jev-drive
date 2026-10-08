#!/usr/bin/env bash
# op_parity turn selector navtrain (plans/2026-10-08-turn-selector-navtrain-prereg.md). One self-advancing chain in tmux jev:
#   scripts/tmux_run.sh turn-selnt experiments/op_parity/scripts/turn_selnt_chain.sh
# extraction (12 shards + navtest, pool fan-out) -> build + G-leak / G-margin -> G-hidden -> pilot (E, P3, N7: G-E already passed, cost gate on N7)
# -> all other arms -> report. State: $DATA_DIR/runs/op_parity/turn_selnt/chain/{STATUS, DONE, ERROR, log.txt}. Rerunning resumes (finished jobs are skipped).
# Waits only on the pool jobs' DONE / ERROR files (no bare `wait`, no pgrep).
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/op_parity/turn_selnt
D=$O/chain; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
CL="$DATA_DIR/envs/jevdrive/bin/python -m jevdrive.cl"
S=experiments/op_parity/scripts
X="$PY $S/turn_selnt.py"
TAG=full
L=$O/pool
status() { echo "$(date '+%F %T') op_parity turn-selnt: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1; shift; local ld=$L/$n; [[ -f $ld/DONE ]] && return
        local live; live=$($CL queue 2>/dev/null | awk -v n="tsn-$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && return; rm -rf "$ld"
        $CL submit --owner op_parity --name "tsn-$n" --log-dir "$ld" "$@" || die "submit $n"; }
waitdirs() { for n in "$@"; do until [[ -f $L/$n/DONE || -f $L/$n/ERROR ]]; do sleep 30; done; [[ -f $L/$n/ERROR ]] && die "job failed: $L/$n/ERROR"; done; return 0; }
GPU_ARMS="N5 N6 N7 N8 P4"; CPU_ARMS="E N1 N3 N4 P1 P3"; TREE_ARMS="N2 P2"

status "extraction: 12 shards + navtest"
for i in $(seq 0 11); do sub ext-$i --vram 16 --cpu 3 --ram 24 -- $PY $S/tsn_extract.py train --shard $i --tag $TAG --control; done
sub ext-navtest --vram 16 --cpu 3 --ram 24 -- $PY $S/tsn_extract.py navtest --tag $TAG
waitdirs $(for i in $(seq 0 11); do echo ext-$i; done) ext-navtest

status "build + G-hidden"
sub build --vram 0.5 --cpu 16 --ram 80 -- $X build --tag $TAG
sub hidden --vram 0.5 --cpu 2 --ram 20 -- $X hidden --tag $TAG
waitdirs build hidden
grep -q '"ok": true' $O/gates_build_$TAG.json || die "build gates failed"

status "pilot: E, P3 (cheap), N7 (cost gate)"
sub fit-E --vram 0.5 --cpu 4 --ram 30 -- $X fit --tag $TAG --arm E
sub fit-P3 --vram 0.5 --cpu 4 --ram 30 -- $X fit --tag $TAG --arm P3
sub fit-N7 --vram 8 --cpu 4 --ram 40 -- $X fit --tag $TAG --arm N7
waitdirs fit-E fit-P3
# cost gate: time of the first N7 config (3 CV fits) x 13.3 = predicted wall; stop above 1.5 x 60 min
for _ in $(seq 1 120); do
  t=$(grep -h "N7 F19 x pc cfg" $L/fit-N7/log.txt 2>/dev/null | head -1 | sed -E 's/.*elapsed ([0-9]+) s.*/\1/')
  [[ -n $t ]] && break; [[ -f $L/fit-N7/ERROR ]] && die "N7 failed"; sleep 30
done
[[ -n ${t:-} ]] || die "N7 did not finish its first config in 60 min"
pred=$(( t * 133 / 10 / 60 )); status "N7 first config ${t}s -> predicted wall ${pred} min"
(( pred > 90 )) && die "N7 predicted wall ${pred} min > 90 min: stop and report (cost gate)"

status "all other arms"
for a in N1 N4 P1; do sub fit-$a --vram 0.5 --cpu 4 --ram 30 -- $X fit --tag $TAG --arm $a; done
sub fit-N3 --vram 0.5 --cpu 4 --ram 30 -- $X fit --tag $TAG --arm N3
for a in $TREE_ARMS; do sub fit-$a --vram 0.5 --cpu 16 --ram 40 -- $X fit --tag $TAG --arm $a; done
for a in N5 N6 N8 P4; do sub fit-$a --vram 8 --cpu 4 --ram 40 -- $X fit --tag $TAG --arm $a; done
waitdirs fit-E fit-P3 fit-N1 fit-N4 fit-P1 fit-N3 fit-N2 fit-P2 fit-N5 fit-N6 fit-N8 fit-P4 fit-N7

status "report"
sub report --vram 0.5 --cpu 8 --ram 40 -- $X report --tag $TAG
waitdirs report
status "done"; date > "$D/DONE"
