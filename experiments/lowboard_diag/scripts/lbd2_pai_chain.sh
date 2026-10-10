#!/usr/bin/env bash
# LOWDIAG2 (prereg amendment 3): the served base stacks P2H10-F-s{2,3} on PAI2's six 10-scene chunks, as FIX1's `ab_*_sN` stacks were run
# (pai_native.sh, JEV_VCONT=1.0 JEV_LEAD=1, one pool job per stack), pruned the same way (zero-score rollout.asl kept for names `ab_*`).
#   scripts/tmux_run.sh lbd2-pai bash experiments/lowboard_diag/scripts/lbd2_pai_chain.sh "2 3"
# State: $DATA_DIR/runs/lowboard_diag/pai2/{STATUS, DONE | ERROR, runs/ab_<chunk>_s<seed>/}. A stack without results is submitted once more.
set -uo pipefail
cd "$(dirname "$0")/../../.."
SEEDS=${1:-"2 3"}; O=$DATA_DIR/runs/lowboard_diag/pai2; S=experiments/alpasim/scripts; CH=experiments/alpasim/results/pai/chunks
mkdir -p "$O/runs"; rm -f "$O/DONE" "$O/ERROR"; VPY=$PWD/.venv/bin/python
ok() { [[ -f $1/aggregate/results-summary.json ]]; }
ended() { ok "$1" || [[ -f $1/pool/ERROR || -f $1/pool/DONE ]]; }
JOBS=(); for s in $SEEDS; do for c in a10 b1 b2a b2b e1 e2; do JOBS+=("ab_${c}_s$s"); done; done
for pass in 1 2; do
  for r in "${JOBS[@]}"; do
    ok "$O/runs/$r" && continue
    IFS=_ read -r _ c s <<< "$r"; R=$O/runs/$r; rm -rf "$R"; mkdir -p "$R/pool"
    $VPY -m jevdrive.cl submit --no-check --owner lowdiag2 --name "pai-lbd2-$r" --vram 24 --cpu 6 --ram 40 --timeout-h 2 --tries 1 --log-dir "$R/pool" -- \
      env CONC=4 SH30_TAG=P2H10-F-${s} JEV_VCONT=1.0 JEV_LEAD=1 bash -c "bash $S/pai_native.sh $R $CH/$c.txt; rc=\$?; python3 $S/fix1_pai_box_report.py prune $R; exit \$rc" | tail -1 >> "$O/jobs.txt"
  done
  while :; do n=0; for r in "${JOBS[@]}"; do ended "$O/runs/$r" && n=$((n + 1)); done; echo "$(date '+%F %T') pass $pass: $n / ${#JOBS[@]} ended" > "$O/STATUS"; (( n == ${#JOBS[@]} )) && break; sleep 60; done
  bad=0; for r in "${JOBS[@]}"; do ok "$O/runs/$r" || bad=$((bad + 1)); done; (( bad == 0 )) && break
done
(( bad == 0 )) && { date > "$O/DONE"; echo done > "$O/STATUS"; } || echo "$bad stacks missing" > "$O/ERROR"
