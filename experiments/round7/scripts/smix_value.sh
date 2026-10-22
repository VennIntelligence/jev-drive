#!/usr/bin/env bash
# SMIX-VALUE chain (box, CPU only, repo root; scripts/tmux_run.sh smixv experiments/round7/scripts/smix_value.sh):
# the CPU-only analyses first, then six stratified S-row forwards three at a time, then learn -> $W/DONE.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=
PY="$DATA_DIR/envs/op-train/bin/python experiments/round7/scripts/smix_value.py"
W=$DATA_DIR/runs/round7/smix_value; mkdir -p $W; rm -f $W/DONE $W/fwd_*.parquet
NC=$(python3 -c "from jevdrive.common import n_cpus; print(n_cpus())"); TH=$(( NC / 3 ))
for c in batch cover pai; do $PY $c > $W/$c.log 2>&1 || echo "$c failed" | tee -a $W/errors.txt; done
fwd() { $PY fwd --ckpt $1 ${2:+--cmd $2} --frac 0.15 --threads $TH > $W/fwd_$1${2:+_cmdh}.log 2>&1 || echo "fwd $1 $2 failed" | tee -a $W/errors.txt; }
fwd R7J-L18-s0 heading & fwd R7J-L18-s1 heading & fwd R6DAG-L18-s0 & wait
fwd R6DAG-L18-s1 & fwd NTV0-L18-s0 & fwd R7J-L18-s1 & wait
$PY learn > $W/learn.log 2>&1 || echo "learn failed" | tee -a $W/errors.txt
date > $W/DONE
