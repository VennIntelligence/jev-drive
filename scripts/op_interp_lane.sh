#!/usr/bin/env bash
# One-shot chain for research/openpilot-openloop-integration.md: WOD ceiling study, then the NAVSIM subset.
# Each step is skipped when its output exists, so a re-run resumes. STATUS / DONE / ERROR in $DATA_DIR/runs/op_interp.
#   scripts/tmux_run.sh opi scripts/op_interp_lane.sh [wod|nav|all]
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd); cd "$repo"
R=$DATA_DIR/runs/op_interp; mkdir -p "$R"
CPUS=${CPUS:-170-171,174-177,190-199}; export CUDA_VISIBLE_DEVICES=${GPU:-2}
OP="taskset -c $CPUS $DATA_DIR/envs/openpilot/bin/python scripts/op_interp.py"
VF="taskset -c $CPUS $DATA_DIR/envs/vfi/bin/python scripts/op_interp.py"
PJ="taskset -c $CPUS $DATA_DIR/envs/jevdrive/bin/python scripts/op_interp.py"
st() { echo "$(date '+%F %T') $*" | tee -a "$R/STATUS"; }
die() { st "ERROR $*"; echo "$*" > "$R/ERROR"; exit 1; }
synth() { local d=$1 m=$2; shift 2; local tag=${TAG:-$m}
  [[ -f $R/$d/$tag.json ]] && return 0
  local py=$OP; [[ $m == rife || $m == gimm ]] && py=$VF
  st "synth $d $tag"; $py synth --data "$d" --method "$m" "$@" || die "synth $d $tag"; }
run() { local d=$1 f=$2 m=$3; shift 3; local tag="$f@$m${SUF:-}"
  [[ -f $R/$d/plans/$tag.npz ]] && return 0
  st "run $d $tag"; $OP run --data "$d" --frames "$f" --model "$m" "$@" || die "run $d $tag"; }
what=${1:-all}
MS=${METHODS:-hold blend warp rife gimm}          # a re-run with the full list resumes

if [[ $what == wod || $what == all ]]; then
  [[ -f $R/wod/meta.json ]] || { st "wod-cache"; $OP wod-cache --workers 16 || die wod-cache; }
  for m in $MS; do synth wod $m; done
  TAG=warp_pre1.5 synth wod warp --preroll 1.5
  TAG=warp_pre3.3 synth wod warp --preroll 3.3
  for mdl in cinque small lebowski; do for f in real $MS; do run wod $f $mdl; done; done
  run wod warp_pre1.5 cinque; run wod warp_pre3.3 lebowski
  SUF=_start-0.5 run wod real cinque --start -0.5
  [[ $MS == *gimm* ]] || { st "wod partial (no gimm)"; exit 0; }
  st "score-wod"; $PJ score-wod --adapters base retime > "$R/wod/score.log" 2>&1 || die score-wod
  st "wod done"
fi

if [[ $what == nav || $what == all ]]; then
  [[ -f $R/nav/meta.json ]] || { st "nav-cache"; $OP nav-cache --n 2000 --workers 16 || die nav-cache; }
  for m in $MS; do synth nav $m; done
  TAG=hold_nominal synth nav hold --keys nominal
  TAG=rife_nominal synth nav rife --keys nominal
  TAG=rife_pre1.5 synth nav rife --preroll 1.5
  for f in $MS hold_nominal rife_nominal rife_pre1.5; do run nav $f cinque; done
  for mdl in small lebowski; do for f in hold warp rife; do run nav $f $mdl; done; done
  SUF=_start-0.5 run nav rife cinque --start -0.5
  st "nav-export"; $PJ nav-export --adapters base || die nav-export
  $PJ nav-export --adapters retime nolever shift cubic --plans hold@cinque rife@cinque || die nav-export-adapters  # adapter ablations
  st "nav-score"; scripts/op_interp_score.sh "$CPUS" > "$R/nav/score_all.log" 2>&1 || die nav-score
  st "nav-report"; $PJ nav-report --refs hold-cinque__base rife-cinque__base > "$R/nav/report.log" 2>&1 || die nav-report
  st "nav done"
fi
st "DONE $what"; touch "$R/DONE_$what"
