#!/usr/bin/env bash
# Night queue 2, executor B: resume after the box restart (todos/2026-09-26-night-queue-2.md, [B] pause entry).
# Everything it restarts is resumable: the per-anchor scoring skips finished chunk files, the devkit jobs rerun only
# the navtest scorings that have no result csv yet, the N4 fit reruns only if its criteria.csv is missing.
#   scripts/night2_b_resume.sh <gpu for the N4 fit>
set -uo pipefail
cd ~/data/jev-drive
GPU=${1:-4}
scripts/tmux_run.sh n3-score-s1 env N3_CPUS=${S1_CPUS:-64-111} scripts/night2_n3_hydra_score.sh 1
scripts/tmux_run.sh n3-score-s2 env N3_CPUS=${S2_CPUS:-160-199} scripts/night2_n3_hydra_score.sh 2
J=$DATA_DIR/runs/night2/n3-navjobs/20260926-101432
: > "$J/score_jobs_resume.txt"
while read -r ver split name path; do
  [[ -z $ver ]] && continue
  ls "$DATA_DIR/runs/navsim/eval/${ver}_${split}_${name}"/*/*.csv >/dev/null 2>&1 || echo "$ver $split $name $path" >> "$J/score_jobs_resume.txt"
done < "$J/score_jobs.txt"
echo "devkit jobs to rerun: $(wc -l < "$J/score_jobs_resume.txt")"
[[ -s $J/score_jobs_resume.txt ]] && scripts/tmux_run.sh n3-score-s0 env NAVSIM_THREADS=8 scripts/real_g1_score.sh "$J/score_jobs_resume.txt" 4
ls "$DATA_DIR"/runs/night2/n4-fit/*/criteria.csv >/dev/null 2>&1 || \
  scripts/tmux_run.sh n4-fit env CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 .venv/bin/python -m jevdrive.night2_n4 fit
