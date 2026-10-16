#!/usr/bin/env bash
# Tokyo only: put the full AlpaSim nuPlan public data under /data/datasets/alpasim_nuplan (box layout).
#   hf   : all asset shards from hf-mirror via fetch_data.sh (sha256-checked tarballs; 4 ranges, 8 draws HTTP 429)
#   prio : the scenes listed in $1 rsynced from the GPU box (`autodl` alias), in list order, size-only skip
#   verify: file/size listing of navtest/assets and nuplan_test against the box; checksum dry run on the priority list
# Usage (Tokyo): bash tokyo_nuplan_full.sh hf|prio <list>|verify <list>. Markers: $OUT/<phase>.DONE / .ERROR
set -euo pipefail
export DATA_DIR=${DATA_DIR:-/data}
ROOT=$DATA_DIR/datasets/alpasim_nuplan; OUT=$DATA_DIR/runs/alpasim/tokyo_full; mkdir -p "$OUT"
BOXA=data/datasets/alpasim_nuplan; ph=$1; rm -f "$OUT/$ph.DONE" "$OUT/$ph.ERROR"
trap 'echo "line $LINENO" > "$OUT/$ph.ERROR"' ERR
case $ph in
hf)  # trajdata cache and configs were fetched on 2026-10-09 (verified there); skip them here, verify below
  touch "$ROOT/.done.trajdata_cache_nuplan_test.tar.gz" "$ROOT/.done.MTGS_asset_navtest_configs.tar.gz"
  JOBS=4 PAR=1 bash "$(dirname "$0")/fetch_data.sh" all ;;
prio)
  mkdir -p "$ROOT/navtest/assets"; t0=$SECONDS
  sed 's|$|/|' "$2" | rsync -a --size-only --partial-dir=.rsync-partial --info=progress2 --files-from=- \
    autodl:$BOXA/navtest/assets/ "$ROOT/navtest/assets/"
  echo "prio $(wc -l < "$2") scenes in $((SECONDS - t0)) s" ;;
verify)
  for d in navtest/assets navtest/configs nuplan_test; do
    (cd "$ROOT/$d" && find . -type f ! -path '*/.rsync-partial/*' -printf '%P %s\n' | sort) > "$OUT/tokyo.$(tr / _ <<<$d).lst"
    ssh autodl "cd ~/$BOXA/$d && find . -type f -printf '%P %s\n' | sort" > "$OUT/box.$(tr / _ <<<$d).lst"
    wc -l "$OUT"/*.$(tr / _ <<<$d).lst; cmp "$OUT/tokyo.$(tr / _ <<<$d).lst" "$OUT/box.$(tr / _ <<<$d).lst"
  done
  sed 's|$|/|' "$2" | rsync -anc --files-from=- --out-format='DIFF %n' autodl:$BOXA/navtest/assets/ "$ROOT/navtest/assets/" | tee "$OUT/prio_checksum_diff.txt"
  [[ ! -s $OUT/prio_checksum_diff.txt ]] ;;
esac
date > "$OUT/$ph.DONE"
