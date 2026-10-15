#!/usr/bin/env bash
# ROUTE-LANE CORRIDOR (plans/2026-10-15-corridor-check-prereg.md): the stages of rf_corridor.py on the two hosts. CPU only, no pool job.
#   Tokyo:  rf_corridor.sh lanes-tokyo <out pkl> <PAI run root> ..   (alpasim-base:0.89.0, niced, CPUS / WORKERS default 4)
#           rf_corridor.sh plans <out pkl> <PAI run root> ..          (host python3)
#   box:    rf_corridor.sh box-prep                                   lanes (held 460), plans and rf_tax classes of the held sets
#           rf_corridor.sh box-run                                    build + check (tmux jev:corridor via scripts/tmux_run.sh)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cmd=$1; shift
case $cmd in
lanes-tokyo)
  OUT=$1; shift; mkdir -p "$(dirname "$OUT")"
  exec nice -n 15 docker run --rm --cpus "${CPUS:-4}" --entrypoint bash \
    -v /data/datasets/nurec:/mnt/nre-data:ro -v /data/third_party/alpasim/src:/repo/src -v /data/third_party/alpasim/plugins:/repo/plugins \
    -v /data/runs:/data/runs:ro -v "$(dirname "$OUT")":/out -v "$HERE/rf_corridor.py":/w/rf_corridor.py:ro -v "$HERE/rf_tax.py":/w/rf_tax.py:ro \
    alpasim-base:0.89.0 -c "umask 0000 && cd /repo && uv run --no-sync python /w/rf_corridor.py lanes --workers ${WORKERS:-4} --out /out/$(basename "$OUT") --runs $*" ;;
plans)
  OUT=$1; shift; exec python3 "$HERE/rf_corridor.py" plans --out "$OUT" --runs "$@" ;;
box-prep)
  O=$DATA_DIR/runs/route_follow/corridor; M=$DATA_DIR/runs/route_mem/read; P=$DATA_DIR/third_party/alpasim/.venv/bin/python
  mkdir -p $O/in $O/tax
  $P "$HERE/rf_corridor.py" lanes --workers 24 --out $O/in/lanes_held.pkl --runs $(ls -d $M/RSG-L18-s0/c*/run) &
  for T in RSG-L18-s0 RSG-L18-s1 RBSG2-L18-s0 RV2c-L18-s0 RV2c-L18-s1 APP1X-L18-s0; do
    python3 "$HERE/rf_corridor.py" plans --out $O/in/plans_$T.pkl --runs $(ls -d $M/$T/c*/run) &
  done
  for T in RBSG2-L18-s0 RV2c-L18-s0 RV2c-L18-s1 APP1X-L18-s0; do      # rf_tax classes of the new extracts (no ref; primaries do not use it)
    .venv/bin/python "$HERE/rf_tax.py" report --set $T=$O/tax/held_$T.pkl --out $O/tax/report_$T > $O/tax/report_$T.log 2>&1 &
  done
  wait; echo box-prep done ;;
box-run)
  O=$DATA_DIR/runs/route_follow/corridor; I=$O/in; L=$DATA_DIR/runs/route_follow; R=experiments/route_follow/results
  S=(pai=$I/pai_rsg.pkl,$I/plans_pai_rsg.pkl,$I/lanes_pai.pkl,$R/taxonomy/pai281_rsg.csv,$R/pai_follow/scenes.csv,rsg)
  for s in 0 1; do S+=(held_RSG_s$s=$L/ledger/held_RSG-L18-s$s.pkl,$I/plans_RSG-L18-s$s.pkl,$I/lanes_held.pkl,$R/ledger/tax/RSG-L18-s$s.csv,$L/report/follow_a3/scenes.csv,RSG-L18-s$s@18); done
  S+=(held_RBSG2_s0=$O/tax/held_RBSG2-L18-s0.pkl,$I/plans_RBSG2-L18-s0.pkl,$I/lanes_held.pkl,$O/tax/report_RBSG2-L18-s0/RBSG2-L18-s0.csv,$L/report/follow_r2/scenes.csv,RBSG2-L18-s0@18)
  for s in 0 1; do S+=(held_RV2c_s$s=$O/tax/held_RV2c-L18-s$s.pkl,$I/plans_RV2c-L18-s$s.pkl,$I/lanes_held.pkl,$O/tax/report_RV2c-L18-s$s/RV2c-L18-s$s.csv,$L/v2/report/follow/scenes.csv,RV2c-L18-s$s@18); done
  S+=(held_APP1X_s0=$O/tax/held_APP1X-L18-s0.pkl,$I/plans_APP1X-L18-s0.pkl,$I/lanes_held.pkl,$O/tax/report_APP1X-L18-s0/APP1X-L18-s0.csv,$L/approach/report/follow/scenes.csv,APP1X-L18-s0@18)
  .venv/bin/python -W ignore "$HERE/rf_corridor.py" build --set "${S[@]}" --out $O/build
  .venv/bin/python -W ignore "$HERE/rf_corridor.py" check --build $O/build --out $R/corridor --set "${S[@]}" --fig experiments/route_follow/figs/corridor_bev.png \
    --picks "${PICKS:-pai:382a7247,pai:de9db5dc,pai:1a69f757,pai:${DONE_PAI:-e848c843}:completed,held_APP1X_s0:${APP_PICK:-x},held_RBSG2_s0:${RB_PICK:-x}}" ;;
*) sed -n '2,7p' "$0"; exit 1 ;;
esac
