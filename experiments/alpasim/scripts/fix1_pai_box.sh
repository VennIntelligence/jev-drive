#!/usr/bin/env bash
# Lane FIX1, PAI track on the GPU box (prereg amendment 3): the serving arms for P2H10-F-s0 / -s1 on lane PAI2's 60 scenes with its chunk
# files (results/pai/chunks/{a10,b1,b2a,b2b,e1,e2}.txt), every stack a pool job (pai_native.sh; --vram 24 --cpu 6 --ram 40), pruned when it
# ends (videos, renderer cache, and rollout.asl of non-zero rollouts are deleted). The base is PAI2's runs ($DATA_DIR/runs/alpasim/pai2/base).
#   scripts/tmux_run.sh fix1-paibox bash experiments/alpasim/scripts/fix1_pai_box.sh
# Step 1: +a+b on chunk b2a, seed 0, alone (the parity run against the Tokyo box); then the grid. A failed stack is submitted once more.
# State: $DATA_DIR/runs/alpasim/fix1/paibox/{STATUS, DONE | ERROR, log.txt, jobs.txt, runs/<arm>_<chunk>_s<seed>/, results/}.
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/alpasim/fix1/paibox; S=experiments/alpasim/scripts; CH=experiments/alpasim/results/pai/chunks; mkdir -p "$O/runs" "$O/results"; rm -f "$O/DONE" "$O/ERROR"
exec > >(tee -a "$O/log.txt") 2>&1
VPY=$PWD/.venv/bin/python
st() { echo "$(date '+%F %T') fix1 paibox: $*" | tee "$O/STATUS"; }
flags() { case $1 in a) echo "JEV_VCONT=1.0" ;; b) echo "JEV_LEAD=1" ;; ab) echo "JEV_VCONT=1.0 JEV_LEAD=1" ;; c1) echo "JEV_BASE=1" ;; c2) echo "JEV_BASE=2" ;;
                    c1b) echo "JEV_BASE=1 JEV_LEAD=1" ;; c2b) echo "JEV_BASE=2 JEV_LEAD=1" ;; esac; }
ok() { [[ -f $1/aggregate/results-summary.json ]]; }
sub() {  # arm chunk seed
  local R=$O/runs/$1_$2_s$3; ok "$R" && return 0
  rm -rf "$R"; mkdir -p "$R/pool"
  # shellcheck disable=SC2046
  $VPY -m jevdrive.cl submit --no-check --owner alpasim-fix1 --name "pai-fix1-$1" --vram 24 --cpu 6 --ram 40 --timeout-h 2 --tries 1 --priority 13 --log-dir "$R/pool" -- \
    env CONC=4 SH30_TAG=P2H10-F-s$3 $(flags "$1") bash -c "bash $S/pai_native.sh $R $CH/$2.txt; rc=\$?; python3 $S/fix1_pai_box_report.py prune $R; exit \$rc" | tail -1 >> "$O/jobs.txt"
}
wait_for() {  # run names...
  local t0=$SECONDS n
  while :; do
    n=0; for r in "$@"; do { ok "$O/runs/$r" || [[ -f $O/runs/$r/pool/ERROR || -f $O/runs/$r/pool/DONE ]]; } && n=$((n + 1)); done
    echo "$(date '+%F %T') fix1 paibox: $n / $# stacks ended, $(df -BG --output=avail "$DATA_DIR" | tail -1 | tr -d ' ') free" > "$O/STATUS"
    (( n == $# )) && break
    (( SECONDS - t0 > 5 * 3600 )) && { st "ERROR timeout"; echo timeout > "$O/ERROR"; exit 1; }
    sleep 60
  done; }
st "code $(git rev-parse --short HEAD); parity run: +a+b on b2a, seed 0"
sub ab b2a 0; wait_for ab_b2a_s0
ok "$O/runs/ab_b2a_s0" || { sub ab b2a 0; wait_for ab_b2a_s0; }
ok "$O/runs/ab_b2a_s0" || { st "ERROR parity run failed twice"; echo parity > "$O/ERROR"; exit 1; }
date > "$O/PARITY_DONE"
ALL=()
for s in 0 1; do for arm in ab a b c2b c1 c2 c1b; do for c in b2a b2b a10 b1 e1 e2; do ALL+=("${arm}_${c}_s$s"); done; done; done
for pass in 1 2; do
  st "grid pass $pass: ${#ALL[@]} stacks"
  for r in "${ALL[@]}"; do IFS=_ read -r arm c s <<< "$r"; sub "$arm" "$c" "${s#s}"; done
  wait_for "${ALL[@]}"
  bad=0; for r in "${ALL[@]}"; do ok "$O/runs/$r" || bad=$((bad + 1)); done; (( bad == 0 )) && break
done
$VPY $S/fix1_pai_box_report.py table --base "$DATA_DIR/runs/alpasim/pai2/base" --runs "$O/runs" --chunks $CH --out "$O/results" > "$O/report.log" 2>&1 || { st "ERROR report (see report.log)"; echo report > "$O/ERROR"; exit 1; }
date > "$O/DONE"; st "done ($bad stacks missing): $O/results/fix1_pai_box.md"
