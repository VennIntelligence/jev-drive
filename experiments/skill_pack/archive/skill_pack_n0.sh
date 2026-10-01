#!/usr/bin/env bash
# Skill pack N0 chain (fc65452:todos/2026-09-29-skill-pack-n0.md): prep -> score T (5 pose files) -> select -> score navtest
# once -> report. CPU only, resumable (each step skips what exists). Writes STATUS / DONE / ERROR in the run dir.
#   scripts/tmux_run.sh sp-n0 experiments/skill_pack/archive/skill_pack_n0.sh
set -uo pipefail
repo=$(cd "$(dirname "$0")/../../.." && pwd); cd "$repo"
CPUS=${SP_CPUS:-155-167}
R=$DATA_DIR/runs/skill_pack/n0; mkdir -p "$R"; rm -f "$R/ERROR"
PY=$DATA_DIR/envs/jevdrive/bin/python
export NAVSIM_THREADS=${NAVSIM_THREADS:-12}
st() { echo "$(date '+%F %T') $*" | tee -a "$R/STATUS"; }
die() { st "ERROR: $*"; echo "$*" > "$R/ERROR"; exit 1; }
T() { taskset -c "$CPUS" nice -n 19 "$@"; }
scored() { ls "$DATA_DIR"/runs/navsim/eval/"$1"/*/*.csv > /dev/null 2>&1; }

[[ -f $R/navtest_cands.npz ]] || { st "prep (CPU scorer refit, candidates)"; T "$PY" -m experiments.skill_pack.archive.skill_pack_n0 prep >> "$R/log.txt" 2>&1 || die prep; }
for f in n1.00 n1.05 n1.10 n1.15 hydra; do
  scored "v1_navtrain_sp_n0_T_$f" && continue
  st "score T $f"
  TOKENS_FILE=$R/T_tokens.txt CACHE_NAME=v1_navtrain_oplb T experiments/zeroshot_openloop/archive/navsim_zs_score.sh score v1 navtrain "sp_n0_T_$f" "$R/T_$f.npz" \
    > "$R/score_T_$f.log" 2>&1 || die "score T $f"
done
[[ -f $R/navtest_n0.npz ]] || { st "select on T"; T "$PY" -m experiments.skill_pack.archive.skill_pack_n0 select >> "$R/log.txt" 2>&1 || die select; }
if ! scored v1_navtest_sp_n0_navtest; then
  st "score navtest (the one test run)"
  T experiments/zeroshot_openloop/archive/navsim_zs_score.sh score v1 navtest sp_n0_navtest "$R/navtest_n0.npz" > "$R/score_navtest.log" 2>&1 || die "score navtest"
fi
st "report"; T "$PY" -m experiments.skill_pack.archive.skill_pack_n0 report > "$R/report.md" 2>> "$R/log.txt" || die report
st "DONE"; touch "$R/DONE"
