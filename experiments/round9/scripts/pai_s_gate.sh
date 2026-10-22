#!/usr/bin/env bash
# ROUND9 PAI-S-GATE chain (box, CPU only, repo root): scripts/tmux_run.sh psg experiments/round9/scripts/pai_s_gate.sh [stage ...]
# default: build + meta + the two R6DAG CPU forwards at once, then gate -> $O/DONE (or $O/FAILED). Stages: build | meta | fwd | gate.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=
O=$DATA_DIR/runs/round9/pai_s_gate; mkdir -p $O; rm -f $O/DONE $O/FAILED
PY=experiments/round9/scripts/pai_s_gate.py; AS=$DATA_DIR/third_party/alpasim; TR=$DATA_DIR/envs/op-train/bin/python
NC=$(python3 -c "from jevdrive.common import n_cpus; print(n_cpus())"); TH=$(( NC / 4 > 1 ? NC / 4 : 1 ))
log() { echo "$(date '+%F %T') $*" | tee -a $O/stages.txt; }
run() { local n=$1; shift; log "$n start"; "$@" > $O/$n.log 2>&1 && log "$n ok" || { log "$n FAILED"; echo $n >> $O/failed.txt; }; }
ST=${@:-build meta fwd gate}; rm -f $O/failed.txt
pre=()
[[ $ST == *build* ]] && { run build env JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD $AS/.venv/bin/python $PY build & pre+=($!); }
[[ $ST == *meta* ]] && { run meta .venv/bin/python $PY meta & pre+=($!); }
if [[ $ST == *fwd* ]]; then
  for c in R6DAG-L18-s1 R6DAG-L18-s0; do run fwd_$c $TR $PY fwd --ckpt $c --threads $TH & pre+=($!); done
fi
wait "${pre[@]}" 2>/dev/null
[[ -s $O/failed.txt ]] && { cp $O/failed.txt $O/FAILED; exit 1; }
[[ $ST == *gate* ]] && run gate $TR $PY gate
[[ -s $O/failed.txt ]] && { cp $O/failed.txt $O/FAILED; exit 1; }
date > $O/DONE
