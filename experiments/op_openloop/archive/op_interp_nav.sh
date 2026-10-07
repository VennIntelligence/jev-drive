#!/usr/bin/env bash
# NAVSIM stage of research/openpilot-diagnosis/index.html, trimmed after the WOD pass (op_interp_lane.sh holds the
# full list): the Cinque ladder, small / Lebowski on hold and RIFE, export, official scoring, report. Resumable.
#   scripts/tmux_run.sh opi-nav experiments/op_openloop/archive/op_interp_nav.sh
set -uo pipefail
repo=$(cd "$(dirname "$0")/../../.." && pwd); cd "$repo"
R=$DATA_DIR/runs/op_interp; CPUS=${CPUS:-170-171,174-177,190-199}; export CUDA_VISIBLE_DEVICES=${GPU:-6}
OP="taskset -c $CPUS $DATA_DIR/envs/openpilot/bin/python experiments/op_openloop/lib/op_interp.py"
PJ="taskset -c $CPUS $DATA_DIR/envs/jevdrive/bin/python experiments/op_openloop/lib/op_interp.py"
st() { echo "$(date '+%F %T') $*" | tee -a "$R/STATUS"; }
die() { st "ERROR $*"; echo "$*" > "$R/ERROR"; exit 1; }
run() { [[ -f $R/nav/plans/$1@$2.npz ]] && return 0; st "run nav $1@$2"; $OP run --data nav --frames "$1" --model "$2" --procs ${PROCS:-3} || die "run nav $1@$2"; }
for f in hold blend warp rife rife_nominal rife_pre1.5 ${EXTRA:-}; do run $f cinque; done
for m in small lebowski; do for f in hold rife; do run $f $m; done; done
st "nav-export"; $PJ nav-export --adapters base || die nav-export
$PJ nav-export --adapters retime nolever shift cubic --plans hold@cinque rife@cinque warp@cinque || die nav-export-adapters
$PJ nav-export --adapters retime --plans rife@small rife@lebowski hold@small hold@lebowski rife_nominal@cinque rife_pre1.5@cinque || die nav-export-retime
st "nav-score"; experiments/op_openloop/archive/op_interp_score.sh "$CPUS" > "$R/nav/score_all.log" 2>&1 || die nav-score
st "nav-report"; $PJ nav-report --refs hold-cinque__base rife-cinque__base > "$R/nav/report.log" 2>&1 || die nav-report
st "nav done"; touch "$R/DONE_nav"
