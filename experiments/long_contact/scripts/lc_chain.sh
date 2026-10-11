#!/usr/bin/env bash
# LONG-PROBE chain (experiments/long_contact/plans/2026-10-11-long-probe-prereg.md): pilot -> ten full runs (5 rungs x 2 seeds), one card at
# a time -> hold-log predictions (+ S0 re-read) -> report. It only submits pool jobs and waits on their DONE / ERROR files.
#   scripts/tmux_run.sh long-probe bash experiments/long_contact/scripts/lc_chain.sh
# State: $DATA_DIR/runs/long_contact/chain/{STATUS, DONE | ERROR, log.txt, jobs.txt}; job logs under chain/pool/<job>/. Rerunning resumes.
# Inputs built before the prereg (labels only): lc_ctx.py run, lc_rows.py gen / priv / survey, lc_native.py.
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/long_contact
D=$O/chain; L=$D/pool; mkdir -p "$L"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
CL="$DATA_DIR/envs/jevdrive/bin/python -m jevdrive.cl"
S=$DATA_DIR/jev-drive/experiments/long_contact/scripts
status() { echo "$(date '+%F %T') long-probe: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
LAST=""
sub() { local n=$1 ld=$2; shift 2; if [[ -f $ld/DONE ]]; then return; fi
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        if [[ -n $live ]]; then LAST=$live; return; fi
        rm -f "$ld/ERROR"
        local aft=(); [[ -n $LAST ]] && aft=(--after "$LAST")
        local id; id=$($CL submit --owner long-probe --name "$n" --log-dir "$ld" "${aft[@]}" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; LAST=$id; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 30; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; return 0; }
rdir() { sed -n 1p "$O/train/runs/$1-s$2.dir"; }
TRAIN=(--train --cpu 6 --ram 28 --timeout-h 2 --tries 2)

status "stage 1: pilot (rungs vis and priv, seed 0, shards s2-s4, 6000 steps; validation logs only)"
for r in vis priv; do
  sub lc-pilot-$r "$L/pilot-$r" "${TRAIN[@]}" --vram 30 -- $PY $S/lc_train.py train --rung $r --seed 0 --shards 2 3 4 --steps 6000 --tag pilot-$r --auto
done
waitdirs "$L/pilot-vis" "$L/pilot-priv"
$PY $S/lc_read.py pilot --runs vis="$(rdir pilot-vis 0)" priv="$(rdir pilot-priv 0)" || die "pilot checklist (prereg section 7): $O/read/pilot.json"

status "stage 2: full runs, 5 rungs x 2 seeds, chained on one card"
for r in blind vis nat priv blind_nat; do for s in 0 1; do
  v=66; [[ $r == blind* ]] && v=8
  sub lc-$r-s$s "$L/full-$r-s$s" "${TRAIN[@]}" --vram $v -- $PY $S/lc_train.py train --rung $r --seed $s --tag full-$r --auto
done; done
waitdirs $(for r in blind vis nat priv blind_nat; do for s in 0 1; do echo "$L/full-$r-s$s"; done; done)

status "stage 3: hold-log predictions (all heads of the ten checkpoints) and the S0 re-read"
RUNS=$(for r in blind vis nat priv blind_nat; do for s in 0 1; do echo -n "$r-s$s=$(rdir full-$r $s) "; done; done)
sub lc-predict "$L/predict" --vram 20 --cpu 6 --ram 40 --timeout-h 2 -- $PY $S/lc_read.py predict --s0 --runs $RUNS
waitdirs "$L/predict"
LAD=$(for r in blind vis nat priv blind_nat; do echo -n "$r=$r-s0,$r-s1 "; done)
sub lc-report "$L/report" --vram 0.5 --cpu 8 --ram 40 --timeout-h 2 -- $PY $S/lc_read.py report --ladder $LAD
waitdirs "$L/report"
status "done: tables in $O/read/report"
echo ok > "$D/DONE"
