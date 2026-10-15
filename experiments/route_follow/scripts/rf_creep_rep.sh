#!/usr/bin/env bash
# CREEP-REP (plans/2026-10-15-creep-replication-prereg.md): the box stages that build the RECEDING-style held extracts of the 17
# checkpoints decision 308 did not use. CPU only, no pool job; run in tmux jev (scripts/tmux_run.sh). Repo root.
#   rf_creep_rep.sh tax       rf_tax extract of the 11 sets without one (WORKERS, default 16, one set at a time) + rf_tax report of all 17
#   rf_creep_rep.sh extract   merged follow rows -> rf_receding.py extract-held of the 13 non-flowhead sets -> $O/held_rep.pkl
#   Mac: copy $O/held_rep.pkl, $DATA_DIR/runs/flowhead/closed_loop/held.pkl and $O/tax/report/*.csv to tmp/creep_rep/, then
#        .venv/bin/python experiments/route_follow/scripts/rf_creep_rep.py read ... (see its docstring)
set -euo pipefail; cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/route_follow/creep_rep; M=$DATA_DIR/runs/route_mem/read; L=$DATA_DIR/runs/route_follow; FH=$DATA_DIR/runs/flowhead/closed_loop
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; RF=experiments/route_follow/scripts
LR="RSGlr03n1250-L18-s0 RSGlr03n2500-L18-s0 RSGlr03n5000-L18-s0 RSGlr10n1250-L18-s0 RSGlr10n5000-L18-s0 RSGlr30n1250-L18-s0 RSGlr30n2500-L18-s0 RSGlr30n5000-L18-s0 RSGlr03n5000-L18-s1"
NEW="$LR APP0C-L18-s0 APP2W05-L18-s0"
pk() { case $1 in RV2c-*) echo $L/corridor/tax/held_$1.pkl ;; FMH-*|RGH-*) echo $FH/tax/held_$1.pkl ;; *) echo $O/tax/held_$1.pkl ;; esac; }
mkdir -p $O/tax/report
case ${1:-} in
tax)
  for T in $NEW; do
    [[ -f $(pk $T) ]] || $P $RF/rf_tax.py extract --runs $(ls -d $M/$T/c*/run) --out $(pk $T) --workers ${WORKERS:-16} > $O/tax/extract_$T.log 2>&1
    echo "$T extracted: $(tail -n 1 $O/tax/extract_$T.log)"
  done
  S=(); for T in $NEW RV2c-L18-s0 RV2c-L18-s1 FMH-F-s0 FMH-F-s1 RGH-F-s0 RGH-F-s1; do S+=("$T=$(pk $T)"); done
  .venv/bin/python $RF/rf_tax.py report --set "${S[@]}" --out $O/tax/report ;;
extract)
  python3 - $O/follow.csv $L/lrsteps/follow/scenes.csv $L/v2/report/follow/scenes.csv $L/approach/w0/report/follow/scenes.csv \
      $L/approach/v2/report/follow/scenes.csv <<'PY'
import csv, sys
want = {f"{t}@18" for t in "RSGlr03n1250-L18-s0 RSGlr03n2500-L18-s0 RSGlr03n5000-L18-s0 RSGlr10n1250-L18-s0 RSGlr10n5000-L18-s0 RSGlr30n1250-L18-s0 RSGlr30n2500-L18-s0 RSGlr30n5000-L18-s0 RSGlr03n5000-L18-s1 APP0C-L18-s0 APP2W05-L18-s0 RV2c-L18-s0 RV2c-L18-s1".split()}
rows, seen = [], set()
for f in sys.argv[2:]:
    for r in csv.DictReader(open(f)):
        if r["label"] in want and (r["label"], r["scene"]) not in seen:
            seen.add((r["label"], r["scene"]))
            rows.append(r)
w = csv.DictWriter(open(sys.argv[1], "w", newline=""), fieldnames=list(rows[0]))
w.writeheader(); w.writerows(rows)
print(len(rows), "follow rows,", len({r["label"] for r in rows}), "labels,", sum(r["turn"] == "True" for r in rows), "turn rows ->", sys.argv[1])
PY
  S=(); for T in $NEW RV2c-L18-s0 RV2c-L18-s1; do S+=("$T=$(pk $T)"); done
  .venv/bin/python $RF/rf_receding.py extract-held --set "${S[@]}" --follow $O/follow.csv --out $O/held_rep.pkl ;;
*) sed -n '2,8p' "$0"; exit 1 ;;
esac
