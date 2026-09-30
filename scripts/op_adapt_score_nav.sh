#!/usr/bin/env bash
# op-adapt r2 package S, NAVSIM chain (cores 0-23): V5a on navtest (navsim2 env), CAM_F0 x per token (jevdrive env),
# offset-slot geometry from the navtrain v1 metric cache (navsim1 env).
# numpy 1.23's bundled OpenBLAS computes wrong pinv / SVD on this host unless the core type is forced (build doc log,
# 2026-09-30 13:5x), so every navsim-env call here sets OPENBLAS_CORETYPE=Haswell.
# Usage (box): scripts/tmux_run.sh s-nav scripts/op_adapt_score_nav.sh [v5 cam extract]
set -uo pipefail
cd "$(dirname "$0")/.."
D=$DATA_DIR/runs/op_adapt_r2/logs/nav
mkdir -p "$D"
rm -f "$D/DONE" "$D/ERROR"
export OPENBLAS_CORETYPE=Haswell PYTHONPATH=$PWD
steps=("$@")
(( ${#steps[@]} )) || steps=(v5 cam extract)
for s in "${steps[@]}"; do
  echo "$(date +%F' '%T) step $s" | tee -a "$D/log.txt"; echo "$s" > "$D/STATUS"
  case $s in
    v5) cmd=("$DATA_DIR/envs/navsim2/bin/python" scripts/op_adapt_nav.py v5 --cache v2_navtest --workers 24) ;;
    cam) cmd=("$DATA_DIR/envs/jevdrive/bin/python" -m jevdrive.op_adapt_score_data nav-cam) ;;
    extract) cmd=("$DATA_DIR/envs/navsim1/bin/python" scripts/op_adapt_nav.py extract --cache v1_navtrain
                  --tokens "$DATA_DIR/runs/op_adapt_r2/offset/tokens.txt" --workers 24) ;;
    *) echo "unknown step $s"; echo "$s" > "$D/ERROR"; exit 2 ;;
  esac
  if ! taskset -c 0-23 "${cmd[@]}" 2>&1 | tee -a "$D/log.txt"; then echo "$s" > "$D/ERROR"; exit 1; fi
done
date +%F' '%T > "$D/DONE"
