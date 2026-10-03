#!/usr/bin/env bash
# Lane EDGE rescale arms (plans/2026-10-04-roadedge-diagnosis-plan.md, addendum 1). CPU only, ~40 cores. Resumes from disk.
# pose files (edge_rescale.py) -> official navtrain v1 PDMS per arm -> per-family pick -> official navtest PDMS + navhard
# two-stage EPDMS of U-own, P-own and the picks -> paired readouts (hist_align_report.py). STATUS / DONE / ERROR in
# $DATA_DIR/runs/skill_pack/edge_diag.
#   scripts/tmux_run.sh edge experiments/skill_pack/scripts/edge_chain.sh
set -uo pipefail
repo=$(cd "$(dirname "$0")/../../.." && pwd); cd "$repo"
R=$DATA_DIR/runs/skill_pack/edge_diag; mkdir -p "$R"; rm -f "$R/ERROR" "$R/DONE"
PROCS=${PROCS:-40}; ARMS=(U-own P-own U1.15 U1.3 U1.45 P1.15 P1.3 P1.45)
NV=$DATA_DIR/envs/navsim2/bin/python; P=$DATA_DIR/runs/op_lb
export OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
st() { echo "$(date '+%F %T') $*" | tee -a "$R/STATUS"; }
die() { st "ERROR: $*"; echo "$*" > "$R/ERROR"; exit 1; }
scored() { ls "$DATA_DIR"/runs/navsim/eval/$1/*/*.csv > /dev/null 2>&1; }
name() { echo "opi_$1_gimm-cinque_edge-$2__base"; }       # name <data> <arm>
score() {   # score <ver> <split> <data> <arm> [subset]
  local n; n=$(name "$3" "$4")
  scored "${1}_${2}_$n" && return 0
  ( [[ -n ${5:-} ]] && export CACHE_NAME=v1_navtrain_oplb TOKENS_FILE=$P/lb_navtrain/tokens.txt
    NAVSIM_THREADS=${T:-13} experiments/zeroshot_openloop/archive/navsim_zs_score.sh score "$1" "$2" "$n" "$P/$3/preds/gimm-cinque_edge-$4__base.npz" > "$R/score_${1}_${2}_$n.log" 2>&1 )
  scored "${1}_${2}_$n" || { st "scoring failed: $1 $2 $n"; return 1; }
  st "scored $1 $2 $n"
}

# 1. pose files
ls "$P"/lb_navhard/preds/gimm-cinque_edge-P1.45__base.npz > /dev/null 2>&1 || { st "pose files"; $NV experiments/skill_pack/scripts/edge_rescale.py > "$R/rescale.log" 2>&1 || die "rescale"; }
# 2. navtrain, all arms
pids=()
for a in "${ARMS[@]}"; do T=5 score v1 navtrain lb_navtrain "$a" subset & pids+=($!); done
for p in "${pids[@]}"; do wait "$p" || die "navtrain scoring"; done
$NV - > "$R/navtrain.log" 2>&1 <<EOF || die "navtrain pick"
import glob, pandas as pd, numpy as np
D = "$DATA_DIR/runs/navsim/eval"
ld = lambda n: (lambda d: d[d.valid.astype(bool) & (d.token != "average")].set_index("token"))(pd.read_csv(sorted(glob.glob(f"{D}/{n}/*/*.csv"))[-1]))
b = ld("v1_navtrain_opi_lb_navtrain_gimm-cinque__base")
best = {}
for a in "${ARMS[*]}".split():
    d = ld("v1_navtrain_opi_lb_navtrain_gimm-cinque_edge-%s__base" % a); i = b.index.intersection(d.index)
    dl = 100 * (d.loc[i, "score"] - b.loc[i, "score"]).mean()
    print(a, len(i), round(100 * d.loc[i, "score"].mean(), 3), round(dl, 3), {c: round(100 * (d.loc[i, c] - b.loc[i, c]).mean(), 3) for c in ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort"]})
    if dl > 0 and dl > best.get(a[0], ("", 0))[1]: best[a[0]] = (a, dl)
print("BASE", round(100 * b.score.mean(), 3), len(b))
print("PICK", " ".join(v[0] for v in best.values()) or "none")
EOF
picks=$(grep '^PICK' "$R/navtrain.log" | cut -d' ' -f2-); st "navtrain picks: $picks"
TEST=(U-own P-own); for p in $picks; do [[ $p != none && ! " ${TEST[*]} " =~ " $p " ]] && TEST+=("$p"); done
echo "${TEST[*]}" > "$R/test_arms.txt"

# 3. test boards (3 jobs x 13 threads at a time)
jobs_=(); for a in "${TEST[@]}"; do jobs_+=("v1 navtest lb_navtest $a" "v2 navhard_two_stage lb_navhard $a"); done
for j in "${jobs_[@]}"; do
  while (( $(jobs -rp | wc -l) >= 3 )); do sleep 20; done
  score $j &
done
wait
for j in "${jobs_[@]}"; do set -- $j; scored "${1}_${2}_$(name "$3" "$4")" || die "official score missing: $j"; done

# 4. paired readouts
full=(); for a in "${TEST[@]}"; do full+=("gimm-cinque_edge-$a"); done
export REPORT_OUT=$repo/experiments/skill_pack/results/roadedge
$NV experiments/skill_pack/scripts/hist_align_report.py navtest --arms "${full[@]}" > "$R/navtest.log" 2>&1 || die "navtest readout"
$NV experiments/skill_pack/scripts/hist_align_report.py navhard --arms "${full[@]}" --procs "$PROCS" > "$R/navhard.log" 2>&1 || die "navhard readout"
st "done"; touch "$R/DONE"
