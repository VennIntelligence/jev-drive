#!/usr/bin/env bash
# op-lb lane (scripts/op_lb.py): frame caches, navtrain-subset metric cache, openpilot `none` rollouts, the
# equivalence check against op_interp's stored navfull / navhard plans, export + official scoring + report.
# Resumable: every step is skipped once its output exists. STATUS / DONE / ERROR / logs in $DATA_DIR/runs/op_lb/lane/.
#
#   scripts/tmux_run.sh op-lb scripts/op_lb_lane.sh [step ...]      steps (default all, in this order):
#     prep mcache gimm warp run compare score report arms
#   arms = navtrain only: Cinque + Lebowski x every non-none desire schedule (allowed on navtrain; test splits wait
#          for the pre-registration), exported and scored like the `none` runs.
# Env: CPUS (taskset list), GPUS (GIMM workers, one per card), RUN_GPU (openpilot sessions), PROCS (ORT sessions),
#      VRAM_GB / CAP_GB (per-process cap / pause threshold of the card's total use), NAVSIM_THREADS (scoring).
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd); cd "$repo"
R=$DATA_DIR/runs/op_lb; L=$R/lane; mkdir -p "$L"
CPUS=${CPUS:-168-183}; GPUS=${GPUS:-"0 1 2 3 4 6"}; RUN_GPU=${RUN_GPU:-6}; PROCS=${PROCS:-8}
VRAM_GB=${VRAM_GB:-12}; CAP_GB=${CAP_GB:-78}
export OPI_ROOT=op_lb CUDA_DEVICE_ORDER=PCI_BUS_ID NAVSIM_THREADS=${NAVSIM_THREADS:-14}
T="nice -n 19 taskset -c $CPUS"
OP="$T $DATA_DIR/envs/openpilot/bin/python scripts/op_lb.py"
VF="$T $DATA_DIR/envs/vfi/bin/python scripts/op_lb.py"
PJ="$T $DATA_DIR/envs/jevdrive/bin/python scripts/op_interp.py"
ALL=(lb_navtest lb_navhard lb_navtrain)
declare -A VER=([lb_navtest]=v1 [lb_navhard]=v2 [lb_navtrain]=v1) SPLIT=([lb_navtest]=navtest [lb_navhard]=navhard_two_stage [lb_navtrain]=navtrain)
st() { echo "$(date '+%F %T') $*" | tee -a "$L/STATUS"; }
die() { st "ERROR $*"; echo "$*" > "$L/ERROR"; exit 1; }
chunks_left() { local d=$1 m=$2 n; n=$(python3 -c "import json;print(len(json.load(open('$R/$d/meta.json'))['names']))")
  echo $(( (n + 31) / 32 - $(ls "$R/$d/$m.chunks" 2>/dev/null | grep -c '\.done$') )); }

step_prep() {
  for d in "${ALL[@]}"; do
    [[ -f $R/$d/meta.json && ( $d != lb_navtrain || -f $R/$d/keys.npy ) ]] && continue
    st "prep $d"; $OP prep --data "$d" --workers 16 || die "prep $d"
  done
}
step_mcache() {   # navtrain subset metric cache, v1.1 devkit (OPENBLAS_CORETYPE=Haswell enforced by navsim_zs_score.sh)
  local c=$DATA_DIR/runs/navsim/metric_cache/v1_navtrain_oplb
  [[ -f $L/mcache.done ]] && return 0
  st "metric cache v1 navtrain subset -> $c"
  TOKENS_FILE=$R/lb_navtrain/tokens.txt CACHE_NAME=v1_navtrain_oplb $T scripts/navsim_zs_score.sh cache v1 navtrain \
    > "$L/mcache.log" 2>&1 || die mcache
  n=$(find "$c" -name metric_cache.pkl | wc -l); st "metric cache: $n pkl"; touch "$L/mcache.done"
}
step_gimm() {
  local left=0 d
  for d in "${ALL[@]}"; do left=$(( left + $(chunks_left "$d" gimm) )); done
  (( left == 0 )) && return 0
  st "gimm: $left chunks on GPUs $GPUS"
  local pids=()
  for g in $GPUS; do
    $VF synth --data "${ALL[@]}" --method gimm --gpu "$g" --vram-gb "$VRAM_GB" --cap-gb "$CAP_GB" > "$L/gimm_gpu$g.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || st "gimm worker $p exited $?"; done
  for d in "${ALL[@]}"; do (( $(chunks_left "$d" gimm) == 0 )) || die "gimm $d incomplete"; done
  st "gimm done"
}
step_warp() {   # navtrain subset only (CPU)
  (( $(chunks_left lb_navtrain warp) == 0 )) && return 0
  st "warp lb_navtrain"; $OP synth --data lb_navtrain --method warp --workers 14 || die warp
}
run() {   # data frames model [schedules...]
  local d=$1 f=$2 m=$3; shift 3
  local s=("${@:-none}") need=0 x
  for x in "${s[@]}"; do
    local stem="$f@$m"; [[ $x != none ]] && stem+=".$(echo "$x" | sed 's/@/_/; s/-/m/g')"
    [[ -f $R/$d/plans/$stem.npz ]] || need=1
  done
  (( need )) || return 0
  st "run $d $f@$m ${s[*]}"
  CUDA_VISIBLE_DEVICES=$RUN_GPU $OP run --data "$d" --frames "$f" --model "$m" --schedule "${s[@]}" --procs "$PROCS" || die "run $d $f $m"
}
step_run() {
  run lb_navtest gimm cinque; run lb_navhard gimm cinque
  run lb_navtrain gimm cinque; run lb_navtrain gimm lebowski; run lb_navtrain warp cinque; run lb_navtrain warp lebowski
}
step_compare() {
  for p in navtest:navfull navhard:navhard; do
    local d=lb_${p%%:*} o=${p##*:}
    $DATA_DIR/envs/jevdrive/bin/python scripts/op_lb.py compare "$R/$d/plans/gimm@cinque.npz" \
      "$DATA_DIR/runs/op_interp/$o/plans/gimm_g0.2@cinque.npz" --out "$R/$d/equivalence.json" | tee -a "$L/STATUS" || die "compare $d"
  done
}
score() {   # data
  local d=$1 sub=()
  [[ $d == lb_navtrain ]] && sub=(SUBSET=1 CACHE_NAME=v1_navtrain_oplb)
  $PJ nav-export --data "$d" --adapters base || die "export $d"
  env "${sub[@]}" scripts/op_interp_score.sh "$CPUS" "$d" "${VER[$d]}" "${SPLIT[$d]}" >> "$R/$d/score.log" 2>&1 || die "score $d"
}
step_score() {
  for d in "${ALL[@]}"; do st "score $d"; score "$d"; done
  if ! ls "$DATA_DIR"/runs/navsim/eval/v1_navtrain_human_oplb/*/*.csv > /dev/null 2>&1; then   # sanity reference
    st "score human on the navtrain subset"
    TOKENS_FILE=$R/lb_navtrain/tokens.txt CACHE_NAME=v1_navtrain_oplb $T scripts/navsim_zs_score.sh score v1 navtrain \
      human_oplb human > "$R/lb_navtrain/score_human.log" 2>&1 || die "score human"
  fi
}
step_report() {
  for d in "${ALL[@]}"; do
    st "report $d"; $PJ nav-report --data "$d" --ver "${VER[$d]}" --split "${SPLIT[$d]}" --refs gimm-cinque__base \
      > "$R/$d/report.log" 2>&1 || die "report $d"
  done
}
step_arms() {
  mapfile -t S < <($DATA_DIR/envs/jevdrive/bin/python -c "import sys; sys.argv=['x']; sys.path.insert(0,'scripts'); import op_lb; print('\n'.join(s for s in op_lb.SCHEDULES if s != 'none'))")
  run lb_navtrain gimm cinque "${S[@]}"; run lb_navtrain gimm lebowski "${S[@]}"
  st "score arms lb_navtrain"; score lb_navtrain; step_report
}

steps=("$@"); (( ${#steps[@]} )) || steps=(prep mcache gimm warp run compare score report arms)
st "start: ${steps[*]} (CPUS $CPUS, GPUS $GPUS, RUN_GPU $RUN_GPU)"
for s in "${steps[@]}"; do "step_$s"; done
st "DONE ${steps[*]}"; touch "$L/DONE"
