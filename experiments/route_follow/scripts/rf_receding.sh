#!/usr/bin/env bash
# RECEDING-TURN fingerprint (plans/2026-10-15-receding-turn-prereg.md): the stages of rf_receding.py on the two hosts. CPU only, no pool job.
#   box:    rf_receding.sh held-tax        rf_tax extract + report of RV2a / RV2b held 460 @18 (RSG reuses route_follow/ledger)
#           rf_receding.sh held-extract    slim per-turn pkl of the six held sets -> $DATA_DIR/runs/route_follow/receding/held.pkl
#   Tokyo:  rf_receding.sh pai-extract     slim per-turn pkl of the four PAI 281 checkpoints (strict 36) -> /data/runs/route_night/receding/pai.pkl
#   Mac:    .venv/bin/python rf_receding.py read --pai pai.pkl --held held.pkl --out results/receding --fig figs/receding_onset.png
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cmd=$1; shift
case $cmd in
held-tax)
  O=$DATA_DIR/runs/route_follow/receding/tax; M=$DATA_DIR/runs/route_mem/read; P=$DATA_DIR/third_party/alpasim/.venv/bin/python
  mkdir -p $O; cd "$HERE/../../.."
  for T in RV2a-L18-s0 RV2a-L18-s1 RV2b-L18-s0 RV2b-L18-s1; do
    ( $P "$HERE/rf_tax.py" extract --runs $(ls -d $M/$T/c*/run) --out $O/held_$T.pkl --workers ${WORKERS:-8} > $O/extract_$T.log 2>&1 &&
      .venv/bin/python "$HERE/rf_tax.py" report --set $T=$O/held_$T.pkl --out $O/report_$T > $O/report_$T.log 2>&1; echo "$T $?" ) &
  done
  wait; echo held-tax done ;;
held-extract)
  O=$DATA_DIR/runs/route_follow/receding; M=$DATA_DIR/runs/route_mem/read; L=$DATA_DIR/runs/route_follow
  S=()
  for T in RSG-L18-s0 RSG-L18-s1; do S+=("$T=$L/ledger/held_$T.pkl"); done
  for T in RV2a-L18-s0 RV2a-L18-s1 RV2b-L18-s0 RV2b-L18-s1; do S+=("$T=$O/tax/held_$T.pkl"); done
  cd "$HERE/../../.."; exec .venv/bin/python "$HERE/rf_receding.py" extract-held --set "${S[@]}" --follow $L/v2/report/follow/scenes.csv --read $M --out $O/held.pkl \
    --zeros "$HERE/../results/progress/zeros.csv" --zmap RSG-L18-s0=held_s0 RSG-L18-s1=held_s1 ;;
pai-extract)
  D=/data/runs/route_night; O=$D/receding; mkdir -p $O
  exec python3 "$HERE/rf_receding.py" extract-pai --strict "$HERE/../results/pai281/strict36.txt" --out $O/pai.pkl \
    --zeros "$HERE/../results/progress/zeros.csv" --zmap RSG-L18-s0=pai --set RSG-L18-s0=$D/pai_follow/r281.pkl:rsg:$D/r281/rsg RV2a-L18-s0=$D/v2pai/RV2a-L18-s0/follow.pkl:rv2a-l18-s0:$D/v2pai/RV2a-L18-s0 \
          RV2b-L18-s1=$D/v2pai/RV2b-L18-s1/follow.pkl:rv2b-l18-s1:$D/v2pai/RV2b-L18-s1 APP1X-L18-s0=$D/v2pai/APP1X-L18-s0/follow.pkl:app1x-l18-s0:$D/v2pai/APP1X-L18-s0 ;;
*) sed -n '2,6p' "$0"; exit 1 ;;
esac
