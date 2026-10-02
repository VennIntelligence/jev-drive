#!/usr/bin/env bash
# Tracker-lag pre-compensation chain, variant MODE (default path; plan addendum 1) (plans/2026-10-04-tracker-precomp-plan.md). CPU only, ~48 cores. Resumes from disk.
# navtrain compensation -> official navtrain PDMS per alpha -> alpha* -> navhard / navtest compensation (native + N4) ->
# official navhard EPDMS + navtest PDMS -> in-process navhard readout (incl. ideal tracker) -> navtest / tracking readouts.
# STATUS / DONE / ERROR in $DATA_DIR/runs/skill_pack/trk.
#   scripts/tmux_run.sh trk experiments/skill_pack/scripts/trk_chain.sh
set -uo pipefail
repo=$(cd "$(dirname "$0")/../../.." && pwd); cd "$repo"
R=$DATA_DIR/runs/skill_pack/trk; mkdir -p "$R"; MODE=${MODE:-path}; rm -f "$R/ERROR_$MODE" "$R/DONE_$MODE"
PROCS=${PROCS:-40}; ALPHAS=(0.25 0.5 0.75 1)
NV=$DATA_DIR/envs/navsim2/bin/python; P=$DATA_DIR/runs/op_lb; N4=$DATA_DIR/runs/skill_pack/raise/n4
export PYTHONPATH=$DATA_DIR/third_party/navsim:$DATA_DIR/third_party/nuplan-devkit OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
st() { echo "$(date '+%F %T') [$MODE] $*" | tee -a "$R/STATUS"; }
die() { st "ERROR: $*"; echo "$*" > "$R/ERROR_$MODE"; exit 1; }
scored() { ls "$DATA_DIR"/runs/navsim/eval/$1/*/*.csv > /dev/null 2>&1; }
score() {   # score <ver> <split> <name> <npz> [subset]
  scored "${1}_${2}_$3" && return 0
  ( unset PYTHONPATH   # navsim_zs_score.sh picks the devkit (v1.1 or v2) itself
    [[ -n ${5:-} ]] && export CACHE_NAME=v1_navtrain_oplb TOKENS_FILE=$P/lb_navtrain/tokens.txt
    NAVSIM_THREADS=${T:-16} experiments/zeroshot_openloop/archive/navsim_zs_score.sh score "$1" "$2" "$3" "$4" > "$R/score_${1}_${2}_$3.log" 2>&1 )
  scored "${1}_${2}_$3" || { st "scoring failed: $1 $2 $3"; return 1; }
  st "scored $1 $2 $3"
}
mk() {   # mk <data> <alphas...> -- <name=npz ...>
  local data=$1; shift; local al=(); while [[ $1 != -- ]]; do al+=("$1"); shift; done; shift
  [[ -f $R/diag_${data}_$MODE.pkl ]] && return 0
  st "compensate $data ($*)"
  $NV experiments/skill_pack/scripts/trk_precomp.py make --data "$data" --alphas "${al[@]}" --src "$@" --modes "$MODE" --procs "$PROCS" >> "$R/make_${data}_$MODE.log" 2>&1 || die "make $data"
}

# 1. navtrain: alpha grid, official v1 PDMS on the 3 000-token subset
mk lb_navtrain "${ALPHAS[@]}" -- native=$P/lb_navtrain/preds/gimm-cinque__base.npz
pids=()
for a in "${ALPHAS[@]}"; do T=6 score v1 navtrain "trk_native_${MODE}_a$a" "$R/lb_navtrain/native_${MODE}_a$a.npz" subset & pids+=($!); done
# 2. test-board compensation meanwhile (pose files for every alpha; which are scored is decided by navtrain)
mk lb_navhard "${ALPHAS[@]}" -- native=$P/lb_navhard/preds/gimm-cinque__base.npz n4=$N4/navhard_n4.npz
mk lb_navtest "${ALPHAS[@]}" -- native=$P/lb_navtest/preds/gimm-cinque__base.npz n4=$N4/navtest_n4.npz
for p in "${pids[@]}"; do wait "$p" || die "navtrain scoring"; done
$NV experiments/skill_pack/scripts/trk_report.py navtrain --mode "$MODE" --alphas "${ALPHAS[@]}" > "$R/navtrain_$MODE.log" 2>&1 || die "navtrain readout"
as=$(grep '^ALPHA_STAR' "$R/navtrain_$MODE.log" | awk '{print $2}')
st "alpha* = $as"
ARMS=(1); [[ $as != none && $as != 1 ]] && ARMS+=("$as")
echo "${ARMS[*]}" > "$R/arms_$MODE.txt"

# 3. official scoring of the test boards (3 jobs x 16 threads at a time)
jobs_=()
for m in native n4; do for a in "${ARMS[@]}"; do
  jobs_+=("v2 navhard_two_stage trk_${m}_${MODE}_a$a $R/lb_navhard/${m}_${MODE}_a$a.npz")
  jobs_+=("v1 navtest trk_${m}_${MODE}_a$a $R/lb_navtest/${m}_${MODE}_a$a.npz")
done; done
for j in "${jobs_[@]}"; do
  while (( $(jobs -rp | wc -l) >= 3 )); do sleep 20; done
  score $j &
done
wait
for j in "${jobs_[@]}"; do set -- $j; scored "${1}_${2}_$3" || die "official score missing: $3 ($2)"; done

# 4. readouts
$NV experiments/skill_pack/scripts/trk_report.py tracking --mode "$MODE" > "$R/tracking_$MODE.log" 2>&1 || die "tracking readout"
$NV experiments/skill_pack/scripts/trk_report.py navtest --mode "$MODE" --alphas "${ARMS[@]}" > "$R/navtest_$MODE.log" 2>&1 || die "navtest readout"
$NV experiments/skill_pack/scripts/trk_report.py navhard --mode "$MODE" --alphas "${ARMS[@]}" --procs "$PROCS" > "$R/navhard_$MODE.log" 2>&1 || die "navhard readout"
st "done"; touch "$R/DONE_$MODE"
