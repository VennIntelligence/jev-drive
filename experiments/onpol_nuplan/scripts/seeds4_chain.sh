#!/usr/bin/env bash
# Lane ONPOL-NUPLAN step 0 (decisions 253 / 254): seeds 2 and 3 of JAPSnoY and APY10m10-AB, trained on the pool and read on all 1491 public nuPlan
# AlpaSim scenes (ap2 driver, lists = the 700 / cf1fresh / ot3new, as joint1_chain.sh stage `arms`), then seeds4_report.py.
#   scripts/tmux_run.sh onpol-seeds4 bash experiments/onpol_nuplan/scripts/seeds4_chain.sh
# State: $DATA_DIR/runs/onpol_nuplan/seeds4/chain/{STATUS, DONE, ERROR, log.txt, train/<job>/}; the loop's own state is seeds4/loop/. Rerunning resumes.
# Estimate: 4 trainings ~35-50 min each in parallel (40 GB, 6 cores each); 20 closed-loop pool jobs (4 tags x 5 lists) of 10-25 min at 4 concurrent
# (OT2_MAX_ACTIVE=4); 2-3 h wall, about 6-7 card-hours.
set -uo pipefail
cd "$(dirname "$0")/../../.."
W=$DATA_DIR/runs/onpol_nuplan/seeds4; D=$W/chain; L=$D/train; mkdir -p "$L"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
VPY=$PWD/.venv/bin/python
CL="$VPY -m jevdrive.cl"
CK=$DATA_DIR/runs/op_parity/runs
status() { echo "$(date '+%F %T') seeds4: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; [[ -n ${LOOP:-} ]] && kill "$LOOP" 2>/dev/null; exit 1; }
sub() { local n=$1; shift; [[ -f $L/$n/DONE ]] && return
        $CL queue 2>/dev/null | awk -v n="j1-$n" '$4 == n && ($2 == "queued" || $2 == "running") {f = 1} END {exit !f}' && return; rm -f "$L/$n/ERROR"
        $CL submit --owner alpasim-onpol-seeds4 --name "j1-$n" --train --priority 15 --vram 40 --cpu 6 --ram 60 --timeout-h 3 --log-dir "$L/$n" -- "$@" > /dev/null || die "submit $n"; }
JT="$PY experiments/alpasim/scripts/joint_train.py train --label-bank --std alpasim"
S="--ho ot1:4,yr1:4,bd4:5 --ho-w 3 --ho-excl navsim/body1-val-logs --agent-lam 10 --shape"
status "code $(git rev-parse --short HEAD); estimate 2-3 h wall, 6-7 card-hours"
TAGS=(JAPSnoY-s2 JAPSnoY-s3 APY10m10-AB-s2 APY10m10-AB-s3)
for t in "${TAGS[@]}"; do
  [[ -f $CK/$t/ckpt-final.pt ]] && continue
  [[ -d $CK/$t && ! -f $L/f-$t/DONE ]] && die "tag $t exists without ckpt-final.pt and without a finished job of this chain: not overwriting"
done
for s in 2 3; do
  sub "f-JAPSnoY-s$s" $JT $S --seed $s --tag "JAPSnoY-s$s"
  sub "f-APY10m10-AB-s$s" $JT --yr-mass 0.1 --seed $s --tag "APY10m10-AB-s$s"
done
status "trainings submitted; closed-loop jobs wait for their checkpoints"
J=()
for t in "${TAGS[@]}"; do J+=("$t:ap2:AP2_TAG=$t:$t" "$t@fresh:ap2:AP2_TAG=$t:$t:cf1fresh" "$t@new:ap2:AP2_TAG=$t:$t:ot3new"); done
OT_LANE=onpol_seeds4 OT_OUT=$W OT_OWNER=alpasim-onpol-seeds4 OT_PRIO=12 OT2_MAX_ACTIVE=4 python3 experiments/alpasim/scripts/ot2_loop.py loop "${J[@]}" &
LOOP=$!
for n in f-JAPSnoY-s2 f-JAPSnoY-s3 f-APY10m10-AB-s2 f-APY10m10-AB-s3; do
  until [[ -f $L/$n/DONE || -f $L/$n/ERROR || -f $CK/${n#f-}/ckpt-final.pt ]]; do sleep 30; done
  [[ -f $L/$n/ERROR ]] && die "training failed: $L/$n"
done
status "trainings done; waiting for the closed-loop jobs"
wait "$LOOP"; rc=$?
[[ $rc -eq 0 ]] || die "ot2_loop rc $rc (see $W/loop/ERROR)"
status "loop done; report"
$VPY experiments/onpol_nuplan/scripts/seeds4_report.py --out "$W/results" > "$D/report.log" 2>&1 || die "report failed (see $D/report.log)"
$CL usage --hours 24 > "$D/usage.txt" 2>&1 || true
date '+%F %T' > "$D/DONE"; status "all done"
