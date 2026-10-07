#!/usr/bin/env bash
# op_parity turn-probe (plans/2026-10-07-turn-probe-prereg.md): one self-advancing chain in tmux jev
# (scripts/tmux_run.sh turn-probe experiments/op_parity/scripts/turn_probe_chain.sh). Every job goes through the pool.
#   small fit (4 000 rows, ridge) -> small report (gate G0) -> full fit (ridge + MLP, 4 arms) -> full report (tables, verdict, figures)
# State: $DATA_DIR/runs/op_parity/turn_probe/chain/{STATUS, DONE, ERROR, log.txt}. Rerunning resumes (finished jobs are skipped).
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/op_parity
D=$O/turn_probe/chain; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
CL="$DATA_DIR/envs/jevdrive/bin/python -m jevdrive.cl"
T=experiments/op_parity/scripts/turn_probe.py
status() { echo "$(date '+%F %T') op_parity turn-probe: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$D/$1; shift; [[ -f $ld/DONE ]] && return; rm -f "$ld/ERROR"
        $CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@" >/dev/null || die "submit $n"
        until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 15; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; }

status "small fit"
sub tp-fit-small --vram 24 --cpu 8 --ram 64 -- $PY $T fit --small
status "small report (gate)"
sub tp-report-small --vram 0.5 --cpu 8 --ram 48 -- $PY $T report --small
# prereg gate: the lane stops only if BOTH arms have T45 skill < 0.15 on the small read (the per-arm G0 of the rule is read on the full run)
$PY -c "import json,sys; s=json.load(open('$O/turn_probe-small/report/verdict.json'))['rules']['ridge']['skill_T45']; print('small-gate skill', s); sys.exit(max(s.values()) < 0.15)" \
    || die "small gate: both arms below 0.15 skill on T45: lane stops"
status "full fit"
sub tp-fit --vram 32 --cpu 8 --ram 96 -- $PY $T fit
status "full report"
sub tp-report --vram 0.5 --cpu 8 --ram 64 -- $PY $T report
status "done"; touch "$D/DONE"
