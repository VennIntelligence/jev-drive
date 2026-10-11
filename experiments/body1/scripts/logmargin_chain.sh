#!/usr/bin/env bash
# Log-consistent margin pilot (experiments/body1/plans/2026-10-11-logmargin-pilot-prereg.md): G0.a identity, the pilot (its G0.b / G0.c run at its
# setup and stop it before the first step if a gate is missed), the readers and the report. Only submits pool jobs and waits on their DONE / ERROR.
#   scripts/tmux_run.sh lm-pilot bash experiments/body1/scripts/logmargin_chain.sh
# State: $DATA_DIR/runs/body1/logmargin/chain/{STATUS, DONE | ERROR, log.txt, jobs.txt}; job logs under chain/pool/<job>/; tables under
# $DATA_DIR/runs/body1/logmargin/ (nothing is written into the checkout). Rerunning resumes (finished jobs are skipped).
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/body1/logmargin
D=$O/chain; L=$D/pool; mkdir -p "$L" "$O/reads"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
CL="$DATA_DIR/envs/jevdrive/bin/python -m jevdrive.cl"
S=experiments/body1/scripts
TAG=P2H10SM-P-s0; REF=P2H10-P-s0; SP=P2H10S-P-s0
HO="--ho ot1:4,yr1:4,bd4:5 --agent-lam 10 --ho-w 3 --ho-excl navsim/body1-val-logs --shape"
status() { echo "$(date '+%F %T') body1 logmargin: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && return; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner body1-logmargin --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 20; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; return 0; }
run_dir() { ls -d "$DATA_DIR/runs/op_parity/train-$1"/* | tail -1; }

status "G0.a: the S recipe with the flag off, 60 steps on shard s2, against the stored B43A7-IDENT-S1"
sub lm-ident "$L/ident" --train --vram 20 --cpu 4 --ram 30 --timeout-h 1 -- \
  $PY $S/bd4_train.py train --seed 0 --data navtrain_full.s2of12 --steps 60 --eval-every 30 $HO --tag LM-IDENT-OFF
waitdirs "$L/ident"
$PY $S/bd4_train.py ident --a "$(run_dir B43A7-IDENT-S1)" --b "$(run_dir LM-IDENT-OFF)" --out "$O/ident_off.json" \
  || die "G0.a missed: the flag-off trainer is not bit-identical to the stored P2H10S code path ($O/ident_off.json)"

status "pilot $TAG (G0.b / G0.c at its setup; exit 2 before any step if a G0 gate is missed)"
sub lm-pilot "$L/pilot" --train --vram 26 --cpu 4 --ram 40 --tries 1 --timeout-h 1 -- \
  $PY $S/bd4_train.py train --seed 0 --data navtrain_full.s2of12 navtrain_full.s3of12 --steps 3000 --eval-every 1000 $HO --logmargin --tag $TAG
waitdirs "$L/pilot"

status "reads: hold / validation-part rates, navtest and hold plan dumps (one read each)"
sub lm-g3-hold "$L/g3-hold" --vram 10 --cpu 2 --ram 20 -- $PY $S/bd4_g3.py --set hold --shards 2 3 --new $TAG --ref $REF $SP --name lm_hold --out "$O/reads"
sub lm-g3-val "$L/g3-val" --vram 10 --cpu 2 --ram 20 -- $PY $S/bd4_g3.py --set val --shards 2 3 --new $TAG --ref $REF $SP --name lm_val --out "$O/reads"
sub lm-dump-navtest "$L/dump-navtest" --vram 16 --cpu 2 --ram 24 -- $PY $S/prog_ol.py states --set navtest --new $TAG $SP --ref $REF --out "$O/reads" --name lm_navtest_w
sub lm-dump-hold "$L/dump-hold" --vram 16 --cpu 2 --ram 28 -- $PY $S/prog_ol.py states --set hold --shards 2 3 --new $TAG $SP --ref $REF --out "$O/reads" --name lm_hold
waitdirs "$L/dump-navtest" "$L/dump-hold"
sub lm-route-navtest "$L/route-navtest" --vram 0.5 --cpu 4 --ram 12 -- $PY $S/route_ol.py --name lm_navtest_w --set navtest --new $TAG $SP --ref $REF --out "$O/reads"
sub lm-route-hold "$L/route-hold" --vram 0.5 --cpu 4 --ram 12 -- $PY $S/route_ol.py --name lm_hold --set hold --shards 2 3 --new $TAG $SP --ref $REF --out "$O/reads"
waitdirs "$L/g3-hold" "$L/g3-val" "$L/route-navtest" "$L/route-hold"

status "report"
sub lm-report "$L/report" --vram 0.5 --cpu 2 --ram 8 -- $PY $S/logmargin_report.py
waitdirs "$L/report"
date '+%F %T' > "$D/DONE"; status "done: $O/report.md"
