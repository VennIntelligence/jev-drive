#!/usr/bin/env bash
# LAUNCH-DIAG (results/rtja/launch_diag.md): why the JTN stage launches late on wait set v2 variant b. Box, repo root; every job through the GPU
# pool (priority PRIO, default 11), ids in $O/jobs.txt. No training: the influence runs keep the weights at the init (--diag-freeze).
#   launch_diag.sh infl <init> [steps]   JTN-L18-s0's exact command (jt.sh train, ARCROWS=brake, DRY) with --init <init>, eager, --diag-every 20
#                                        --diag-infl 128 --diag-freeze, <steps> (300) -> runs/route_mem/train-LDF-<init>/ (scalars infl/*)
#   launch_diag.sh replay <dump> <tag ..> [after]   launch_diag.py replay (GPU) of the tags on <dump>'s dumped inputs
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/defect_fix/launch_diag; TR=$DATA_DIR/envs/op-train/bin/python; ME=experiments/defect_fix/scripts/launch_diag.sh
mkdir -p $O/pool
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner defect-fix --name "$name" --tries 1 --priority ${PRIO:-11} \
     --no-check --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
case ${1:-} in
infl)
  I=${2:?init}; N=${3:-300}
  sub ld-infl${SFX:-}-$I --train --vram 56 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=rm-train --env JEV_RM_EAGER=1 ${SFX:+--env SFX=$SFX} ${DE:+--env DE=$DE} -- bash $ME infl-body $I $N ;;
infl-body)
  I=$2; N=$3; T=LDF${SFX:-}-$I
  CMD=$(DRY=1 ARCROWS=brake bash experiments/defect_fix/scripts/jt.sh train $T on1_JT $DATA_DIR/runs/defect_fix/lat/L18NR1v2 JT00 $N 0 | tail -1)
  [[ $CMD == *"--init RB0-F-s0 "* ]] || die "unexpected JTN command: $CMD"
  CMD=${CMD/--init RB0-F-s0 /--init $I }
  echo "$(date '+%F %T') infl $T: $CMD --diag-every ${DE:-20} --diag-infl 128 --diag-freeze" | tee -a $O/stages.txt
  exec $CMD --diag-every ${DE:-20} --diag-infl 128 --diag-freeze ;;
replay)
  D=${2:?dump}; shift 2; A=""; tags=()
  for x in "$@"; do [[ $x == 1018-* || $x == 10[0-9][0-9]-* ]] && A=$x || tags+=($x); done
  sub ld-replay-$D ${A:+--after $A} --vram 16 --cpu 4 --ram 24 --timeout-h 2 -- $TR experiments/defect_fix/scripts/launch_diag.py replay --dump $D --tags ${tags[@]} ;;
*) sed -n 2,7p $0; exit 2 ;;
esac
