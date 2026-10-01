#!/usr/bin/env bash
# GIMM-VFI on the context-rate grid only (t0 - 0.2 k): equivalence check on WOD against the full 10 Hz GIMM feed, then
# the NAVSIM subset (research/openpilot-openloop-integration.md). Resumable; GPU 6, batch 4 (~6 GB).
set -uo pipefail
repo=$(cd "$(dirname "$0")/../../.." && pwd); cd "$repo"
R=$DATA_DIR/runs/op_interp; CPUS=${CPUS:-110-117,184-189}; export CUDA_VISIBLE_DEVICES=${GPU:-6}
OP="taskset -c $CPUS $DATA_DIR/envs/openpilot/bin/python experiments/op_openloop/lib/op_interp.py"
VF="taskset -c $CPUS $DATA_DIR/envs/vfi/bin/python experiments/op_openloop/lib/op_interp.py"
st() { echo "$(date '+%F %T') $*" | tee -a "$R/STATUS"; }
die() { st "ERROR $*"; exit 1; }
for d in wod nav; do
  [[ -f $R/$d/gimm_g0.2.json ]] || { st "synth $d gimm_g0.2"; $VF synth --data $d --method gimm --grid 0.2 --batch 4 --workers 12 || die "synth $d gimm_g0.2"; }
  [[ -f $R/$d/plans/gimm_g0.2@cinque.npz ]] || { st "run $d gimm_g0.2@cinque"; $OP run --data $d --frames gimm_g0.2 --model cinque --procs 3 || die "run $d gimm_g0.2"; }
done
st "gimm grid done"
