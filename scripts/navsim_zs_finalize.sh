#!/usr/bin/env bash
# NAVSIM zero-shot: score everything that is not scored yet, as soon as its predictions exist
# (todos/2026-09-24-zeroshot-exam/navsim.md). Usage: scripts/navsim_zs_finalize.sh alpamayo|openpilot
#   alpamayo   wait for every Alpamayo process, collect the no-nav subset + navhard nav, score them (v1 + v2)
#   openpilot  wait for every openpilot run, score small / Cinque (navtest v1 + v2) and all three on navhard
set -uo pipefail
cd "$(dirname "$0")/.."
S() { NAVSIM_THREADS=${T:-16} scripts/navsim_zs_score.sh score "$@"; }
R=$DATA_DIR/runs/navsim_zs
PY=$DATA_DIR/envs/navsim2/bin/python
case $1 in
alpamayo)
  while pgrep -u "$USER" -f "[n]avsim_zs_alpamayo.py run" >/dev/null; do sleep 60; done
  $PY -c "import sys; sys.path.insert(0, '.'); from jevdrive import navsim_zs as Z
print(Z.collect_alpamayo('navtest', 'nonav', 'repeat')); print(Z.collect_alpamayo('navhard_two_stage', 'nav', 'repeat'))
open('$R/preds/navtest/nonav3k_tokens.txt', 'w').write(chr(10).join(sorted(Z.nonav_subset(Z.load_index('navtest')))) + chr(10))" || exit 1
  S v2 navhard_two_stage alpamayo_nav $R/preds/navhard_two_stage/alpamayo_nav_repeat_main.npz
  export TOKENS_FILE=$R/preds/navtest/nonav3k_tokens.txt   # the no-nav predictions exist only for this subset
  S v2 navtest alpamayo_nonav $R/preds/navtest/alpamayo_nonav_repeat_main.npz
  S v1 navtest alpamayo_nonav $R/preds/navtest/alpamayo_nonav_repeat_main.npz ;;
openpilot)
  while pgrep -u "$USER" -f "[n]avsim_zs_openpilot.py run" >/dev/null; do sleep 60; done
  for m in lebowski cinque small; do for d in none cmd; do
    S v2 navhard_two_stage ${m}_$d $R/openpilot/navhard_two_stage/${m}_$d.npz
    [[ $m == lebowski ]] && continue
    S v2 navtest ${m}_$d $R/openpilot/navtest/${m}_$d.npz
    S v1 navtest ${m}_$d $R/openpilot/navtest/${m}_$d.npz
  done; done ;;
esac
echo "finalize $1 done"
