#!/usr/bin/env bash
# ROUND9 HINGE-AUDIT remainder (box, repo root; scripts/tmux_run.sh ha_rest bash experiments/round9/scripts/hinge_audit_rest.sh): wait for the forwards, steps S / PAIH (PAID runs in jev:ha_paid), report, g5 -> $O/DONE_rest
set -uo pipefail
O=$DATA_DIR/runs/round9/hinge_audit; TR=$DATA_DIR/envs/op-train/bin/python; PY=experiments/round9/scripts/hinge_audit.py
NC=$(python3 -c "from jevdrive.common import n_cpus; print(n_cpus())"); TH=$(( NC / 4 > 1 ? NC / 4 : 1 ))
until [ -f $O/DONE_fwd ] || [ -f $O/FAILED ]; do sleep 30; done
[ -f $O/FAILED ] && exit 1
echo "$(date '+%F %T') rest start" >> $O/stages.txt
$TR $PY steps --pool S --threads $TH > $O/steps_S.log 2>&1 &
$TR $PY steps --pool PAIH --threads $TH > $O/steps_PAIH.log 2>&1 &
wait
until grep -q "^written" $O/steps_PAID.log 2>/dev/null; do sleep 30; done
$TR $PY report > $O/report.log 2>&1 || { echo report >> $O/FAILED; exit 1; }
for p in S PAID PAIH; do $TR $PY g5 --pool $p --threads $TH > $O/g5_$p.log 2>&1 & done
wait
date > $O/DONE_rest
