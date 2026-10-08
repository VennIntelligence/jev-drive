#!/usr/bin/env bash
# op_parity off-track rows (plans/2026-10-09-offtrack-rows-prereg.md): the SH30 recipe with ~10 % statically perturbed (re-projected) rows.
# One self-advancing chain per stage in tmux jev (scripts/tmux_run.sh ot-rows experiments/op_parity/scripts/ot_rows_chain.sh <stage>). Every GPU /
# CPU job goes through the pool, every benchmark read through jevdrive.bench. Rerunning a stage resumes.
#   smoke   64 rows of one shard: zero-offset tokens = the stored W tokens (pipeline gate), perturbed rows, sign probe with SHP-F-s0, timing
#   run     pilot shards (2-4) prepared -> OTP-F-s0 (10 % off-track rows) + OTC-F-s0 (0 %, trainer check), 3 000 steps -> navhard (G) / navtest,
#           HUGSIM 64 guardrail and the AlpaSim-standard offline read on the pilot checkpoint -> GATE (navhard combined >= +1.0 vs the pilot
#           reference recipe; else GATE_STOP) -> the other 9 shards -> OT30-F-s0 / s1 (10 000 x 128) -> navhard (G) / navtest / HUGSIM 64 /
#           offline read -> reports
# State: $DATA_DIR/runs/op_parity/ot_rows/chain-<stage>/{STATUS, DONE, ERROR, GATE_STOP, log.txt, jobs.txt}; results under .../ot_rows/results/.
set -uo pipefail
cd "$(dirname "$0")/../../.."
STAGE=${1:-run}
O=$DATA_DIR/runs/op_parity/ot_rows
D=$O/chain-$STAGE; mkdir -p "$D" "$O/results"; rm -f "$D/DONE" "$D/ERROR" "$D/GATE_STOP"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
VPY=$PWD/.venv/bin/python
CL="$VPY -m jevdrive.cl"
S=experiments/op_parity/scripts
SA=experiments/alpasim/scripts
B=("$VPY" -m jevdrive.bench)
L=$O/pool
R=$O/results
A=$DATA_DIR/runs/alpasim/ap2
K=12
PILOT="navtrain_full.s2of12 navtrain_full.s3of12 navtrain_full.s4of12"
REST=$(for i in 0 1 5 6 7 8 9 10 11; do echo -n "navtrain_full.s${i}of$K "; done)
ALL=$(for i in $(seq 0 $((K - 1))); do echo -n "navtrain_full.s${i}of$K "; done)
PILOT_SPLIT=navsim/op-parity-s234 FULL_SPLIT=navsim/op-parity-full
LIST=experiments/hugsim/scripts/derot_all64.txt
SCENES=$DATA_DIR/runs/alpasim/scenes_navtest_full_part001_48.txt
HP=spec_plan_smooth
status() { echo "$(date '+%F %T') op_parity ot-rows $STAGE: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return
        $CL queue 2>/dev/null | awk -v p="$ld/log.txt" '($2 == "queued" || $2 == "running" || $2 == "inbox") && index($0, p) {f = 1} END {exit !f}' && return
        rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity-r1 --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 20; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; return 0; }
prep() { local d=$1; shift; sub ot-prep $L/prep-$d${TAGX:-} --vram 12 --cpu 20 --ram 60 -- $PY $S/ot_rows.py prep --data $d --workers 18 "$@"; }
offline() {  # name ref keys-regex models... : AlpaSim-standard plans on navtest (m = 1..4), devkit scores of the 2 000-token subset, report
  local name=$1 ref=$2 kre=$3; shift 3
  sub ot-ap2-offline $L/offline-$name --vram 20 --cpu 8 --ram 60 -- $PY $SA/ap2_offline.py plans --name $name --scenes $SCENES --models "$@"
  waitdirs $L/offline-$name
  local keys; keys=$($VPY -c "import numpy as np, re, sys; print(' '.join(k for k in np.load(sys.argv[1]).files if re.search(sys.argv[2], k)))" $A/offline/$name/poses.npz "$kre")
  [[ -f $A/offline/$name/scores.csv ]] || "${B[@]}" score-poses --poses $A/offline/$name/poses.npz --keys $keys --tokens $A/offline/$name/subset.txt \
      --out $A/offline/$name/scores.csv --traffic non_reactive --owner op_parity-r1 --wait || die "score-poses $name"
  $PY $SA/ap2_offline.py report --name $name --ref $ref --scores $A/offline/$name/scores.csv --out $R/ap2_offline_$name || die "offline report $name"
}

if [[ $STAGE == smoke ]]; then
  d=navtrain_full.s2of12
  status "64 rows: zero-offset equality, perturbed rows, sign probe"
  TAGX=-zero64 prep $d --limit 64 --zero
  TAGX=-first64 prep $d --limit 64
  waitdirs $L/prep-$d-zero64 $L/prep-$d-first64
  sub ot-probe $L/probe-smoke --vram 16 --cpu 4 --ram 30 -- $PY $S/ot_rows.py probe --name smoke --tags P0 SHP-F-s0 --data $d --suffix -first64 --min-slope 0.3
  waitdirs $L/probe-smoke
  cat $O/probe_smoke.md; cat "$DATA_DIR/runs/op_parity/cache/ot1_$d-first64@warp/timing.json"
  status "done"; date > "$D/DONE"; exit 0
fi
[[ $STAGE == run ]] || die "unknown stage $STAGE"

# ---------------------------------------------------------------- 1. pilot data; reference reads start at once (their inputs exist)
status "pilot: off-track rows of shards 2-4; reference navhard / HUGSIM reads"
for d in $PILOT; do prep $d; done
"${B[@]}" run --model SHP-F-s0@gimm SHP-F-s1@gimm --bench navhard || die "bench navhard SHP"
"${B[@]}" run --model SHP-F-s0 --bench hugsim --preset $HP --scenarios "$LIST" || die "bench hugsim SHP"
waitdirs $(for d in $PILOT; do echo $L/prep-$d; done)
sub ot-probe $L/probe-pilot0 --vram 16 --cpu 4 --ram 40 -- $PY $S/ot_rows.py probe --name pilot_ref --tags P0 SHP-F-s0 --data $PILOT --split $PILOT_SPLIT --min-slope 0.3

# ---------------------------------------------------------------- 2. pilot arms (seed 0): 10 % off-track rows, and 0 % in the same loop
T="$PY $S/ot_rows.py train --data $PILOT --split $PILOT_SPLIT --steps 3000 --batch 64 --warmup 100 --eval-every 1000 --seed 0"
status "pilot: train OTP-F-s0 / OTC-F-s0"
sub ot-t-pilot $L/t-OTP-s0 --train --vram 24 --cpu 6 --ram 40 -- $T --tag OTP-F-s0 --ot-mass 0.1
sub ot-t-pilot $L/t-OTC-s0 --train --vram 24 --cpu 6 --ram 40 -- $T --tag OTC-F-s0 --ot-mass 0
waitdirs $L/probe-pilot0 $L/t-OTP-s0 $L/t-OTC-s0
for t in OTP OTC; do $PY $S/pp_full_check.py train --tag $t-F-s0 || die "training sanity $t-F-s0"; done
status "pilot: navhard (G) / navtest / HUGSIM guardrail / offline read"
"${B[@]}" run --model OTP-F-s0@gimm OTC-F-s0@gimm --bench navhard || die "bench navhard pilot"
"${B[@]}" run --model OTP-F-s0 OTC-F-s0 --bench navtest || die "bench navtest pilot"
"${B[@]}" run --model OTP-F-s0 --bench hugsim --preset $HP --scenarios "$LIST" || die "bench hugsim OTP"
sub ot-probe $L/probe-pilot1 --vram 16 --cpu 4 --ram 40 -- $PY $S/ot_rows.py probe --name pilot --tags SHP-F-s0 OTC-F-s0 OTP-F-s0 --data $PILOT --split $PILOT_SPLIT
"${B[@]}" status --model OTP-F-s0@gimm OTC-F-s0@gimm SHP-F-s0@gimm SHP-F-s1@gimm --bench navhard --wait || die "navhard pilot"
$VPY $S/ot_rows.py gate --new OTP-F-s0 --refs SHP-F-s0 SHP-F-s1 OTC-F-s0 > "$D/gate.txt" 2>&1; rc=$?
tail -n 12 "$D/gate.txt"
(( rc == 0 || rc == 2 )) || die "gate"
(( rc == 2 )) && cp "$O/gate.json" "$D/GATE_STOP"
pilot_reports() {
  "${B[@]}" status --model OTP-F-s0 OTC-F-s0 --bench navtest --wait || die "navtest pilot"
  "${B[@]}" status --model OTP-F-s0 SHP-F-s0 --bench hugsim --preset $HP --wait || die "hugsim pilot"
  waitdirs $L/probe-pilot1
  "${B[@]}" report --bench navhard --arms OTP=OTP-F-s0@gimm --vs REF=SHP-F-s0@gimm+SHP-F-s1@gimm+OTC-F-s0@gimm SHP0=SHP-F-s0@gimm SHP1=SHP-F-s1@gimm OTC=OTC-F-s0@gimm \
      --out $R/pilot || die "report navhard pilot"
  "${B[@]}" report --bench navtest --arms OTP=OTP-F-s0 --vs REF=SHP-F-s0+SHP-F-s1+OTC-F-s0 SHP0=SHP-F-s0 OTC=OTC-F-s0 --out $R/pilot || die "report navtest pilot"
  "${B[@]}" report --bench hugsim --preset $HP --arms OTP=OTP-F-s0 --vs SHP0=SHP-F-s0 SH30=SH30-F-s0+SH30-F-s1 --scenarios all64 --out $R/pilot || die "report hugsim pilot"
  $VPY $S/ot_rows.py report --arms OTP-F-s0 --refs SHP-F-s0 SHP-F-s1 OTC-F-s0 --out $R/pilot/verdict || die "pilot verdict table"
  offline ot-pilot SHP '_m[14]$' SHP=SHP-F-s0 OTP=OTP-F-s0
  cp $O/probe_pilot_ref.md $O/probe_pilot.md $O/gate.json $R/pilot/ 2>/dev/null
}
if (( rc == 2 )); then
  status "pilot gate: STOP (navhard point estimate < +1.0); writing the pilot reports"
  pilot_reports; status "done (gate stop)"; date > "$D/DONE"; exit 0
fi
status "pilot gate passed; full scale"

# ---------------------------------------------------------------- 3. full recipe x 2 seeds (the pilot reports are written while it trains)
for d in $REST; do prep $d; done
waitdirs $(for d in $REST; do echo $L/prep-$d; done)
TF="$PY $S/ot_rows.py train --data $ALL --split $FULL_SPLIT --batch 128 --warmup 300 --eval-every 1000 --ot-mass 0.1"
sub ot-t-smoke $L/t-smoke-full --train --vram 40 --cpu 6 --ram 80 -- $TF --steps 20 --eval-every 20 --seed 0 --tag smoke-ot30
waitdirs $L/t-smoke-full
for s in 0 1; do sub ot-t-full $L/t-OT30-s$s --train --vram 40 --cpu 8 --ram 80 -- $TF --steps 10000 --seed $s --tag OT30-F-s$s; done
pilot_reports
waitdirs $L/t-OT30-s0 $L/t-OT30-s1
for s in 0 1; do $PY $S/pp_full_check.py train --tag OT30-F-s$s || die "training sanity OT30-F-s$s"; done
status "full: navhard (G) / navtest / HUGSIM 64 / offline read"
"${B[@]}" run --model OT30-F-s0@gimm OT30-F-s1@gimm --bench navhard || die "bench navhard"
"${B[@]}" run --model OT30-F-s0 OT30-F-s1 --bench navtest || die "bench navtest"
"${B[@]}" run --model OT30-F-s0 OT30-F-s1 --bench hugsim --preset $HP --scenarios "$LIST" || die "bench hugsim"
sub ot-probe $L/probe-full --vram 16 --cpu 4 --ram 40 -- $PY $S/ot_rows.py probe --name full --tags SH30-F-s0 SH30-F-s1 OT30-F-s0 OT30-F-s1 --data $PILOT --split $FULL_SPLIT
"${B[@]}" status --model OT30-F-s0@gimm OT30-F-s1@gimm --bench navhard --wait || die "navhard"
"${B[@]}" status --model OT30-F-s0 OT30-F-s1 --bench navtest --wait || die "navtest"
$VPY $S/ot_rows.py report --arms OT30-F-s0 OT30-F-s1 --refs SH30-F-s0 SH30-F-s1 --out $R/full/verdict || die "verdict table"
"${B[@]}" report --bench navhard --arms OT30=OT30-F-s0@gimm+OT30-F-s1@gimm --vs SH30=SH30-F-s0@gimm+SH30-F-s1@gimm WA-JEPA --out $R/full || die "report navhard"
"${B[@]}" report --bench navtest --arms OT30=OT30-F-s0+OT30-F-s1 --vs SH30=SH30-F-s0+SH30-F-s1 WA-JEPA --out $R/full || die "report navtest"
offline ot-full SH30 '^(SH30|OT30)_m[1-4]$' SH30=SH30-F-s0 OT30=OT30-F-s0 OT30s1=OT30-F-s1
"${B[@]}" status --model OT30-F-s0 OT30-F-s1 --bench hugsim --preset $HP --wait || die "hugsim"
"${B[@]}" report --bench hugsim --preset $HP --arms OT30=OT30-F-s0+OT30-F-s1 --vs SH30=SH30-F-s0+SH30-F-s1 --scenarios all64 --out $R/full || die "report hugsim"
waitdirs $L/probe-full; cp $O/probe_full.md $R/full/ 2>/dev/null
status "done"; date > "$D/DONE"
