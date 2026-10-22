#!/usr/bin/env bash
# ROUND9 HINGE-AUDIT chain (box, CPU only, repo root): scripts/tmux_run.sh ha experiments/round9/scripts/hinge_audit.sh [stage ...]
# Stages: fwd (R6DAG-L18-s1 CPU forwards: nuPlan S rows; PAI human rows + navtrain, in parallel) | steps (per-step clearances of every
# pool and path) | report. Writes $O/DONE_<stages> (or $O/FAILED).
set -uo pipefail
export CUDA_VISIBLE_DEVICES=
O=$DATA_DIR/runs/round9/hinge_audit; mkdir -p $O; rm -f $O/FAILED
PY=experiments/round9/scripts/hinge_audit.py; TR=$DATA_DIR/envs/op-train/bin/python
NC=$(python3 -c "from jevdrive.common import n_cpus; print(n_cpus())"); TH=$(( NC / 3 > 1 ? NC / 3 : 1 ))
log() { echo "$(date '+%F %T') $*" | tee -a $O/stages.txt; }
run() { local n=$1; shift; log "$n start"; "$@" > $O/$n.log 2>&1 && log "$n ok" || { log "$n FAILED"; echo $n >> $O/failed.txt; }; }
ST=${@:-fwd steps report}; rm -f $O/failed.txt
pre=()
if [[ $ST == *fwd* ]]; then
  run fwd_S $TR $PY fwd --parts S --frac 0.3 --threads $TH & pre+=($!)
  ( run fwd_ph $TR $PY fwd --parts ph --frac 0.06 --threads $TH && run fwd_nav $TR $PY fwd --parts nav --frac 0.05 --threads $TH ) & pre+=($!)
fi
wait "${pre[@]}" 2>/dev/null; pre=()
if [[ $ST == *steps* ]]; then
  for p in S PAID PAIH; do run steps_$p $TR $PY steps --pool $p --threads $(( NC / 4 > 1 ? NC / 4 : 1 )) & pre+=($!); done
fi
wait "${pre[@]}" 2>/dev/null
[[ $ST == *report* ]] && run report $TR $PY report
[[ -s $O/failed.txt ]] && { cp $O/failed.txt $O/FAILED; exit 1; }
date > "$O/DONE_${ST// /_}"
