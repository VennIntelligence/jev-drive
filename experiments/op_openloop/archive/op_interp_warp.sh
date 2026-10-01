#!/usr/bin/env bash
# NAVSIM follow-up after the first readout (warp was the best feed): warp with pre-roll for Cinque (1.6 s context) and
# Lebowski (4.8 s), warp for small; export (base + retime) and score. Starts once the GIMM chain is done (GPU 6 <= 15 GB).
set -uo pipefail
repo=$(cd "$(dirname "$0")/../../.." && pwd); cd "$repo"
R=$DATA_DIR/runs/op_interp; CPUS=${CPUS:-110-117,184-189}; export CUDA_VISIBLE_DEVICES=${GPU:-6}
OP="taskset -c $CPUS $DATA_DIR/envs/openpilot/bin/python experiments/op_openloop/lib/op_interp.py"
VF="taskset -c $CPUS $DATA_DIR/envs/vfi/bin/python experiments/op_openloop/lib/op_interp.py"
PJ="taskset -c $CPUS $DATA_DIR/envs/jevdrive/bin/python experiments/op_openloop/lib/op_interp.py"
st() { echo "$(date '+%F %T') $*" | tee -a "$R/STATUS"; }
die() { st "ERROR $*"; exit 1; }
for p in 1.5 3.3; do [[ -f $R/nav/warp_pre$p.json ]] || { st "synth nav warp_pre$p"; $VF synth --data nav --method warp --preroll $p --workers 14 || die "synth warp_pre$p"; }; done
until grep -q "gimm grid done" "$R/STATUS"; do sleep 30; done
run() { [[ -f $R/nav/plans/$1@$2.npz ]] && return 0; st "run nav $1@$2"; $OP run --data nav --frames "$1" --model "$2" --procs 2 || die "run nav $1@$2"; }
run warp_pre1.5 cinque; run warp_pre3.3 lebowski; run warp small; run warp lebowski; run gimm_g0.2 cinque
$PJ nav-export --adapters base retime --plans warp_pre1.5@cinque warp_pre3.3@lebowski warp@small warp@lebowski gimm_g0.2@cinque || die export
NAVSIM_THREADS=14 experiments/op_openloop/archive/op_interp_score.sh "$CPUS" > "$R/nav/score_warp.log" 2>&1 || die score
st "warp follow-up done"
