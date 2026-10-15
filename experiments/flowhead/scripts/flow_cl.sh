#!/usr/bin/env bash
# FLOW1 in closed loop (experiments/route_follow/plans/2026-10-15-mode-averaging-prereg.md, Reading 2): the flow head FMH-F-s{0,1} and its
# regression twin RGH-F-s{0,1} served by the AlpaSim sh30 driver (sh30_core.py serves thead.pt; a flow head logs its 8 samples as `th_k`).
# Every GPU job through the pool; ids appended to $O/jobs.txt. Box, repo root.
#   flow_cl.sh submit [tags]   per tag: G-eq (sh30_check parity on 300 navtest tokens) -> gate verdict -> held 460 @18 read (rm_chain.sh read, 6 chunks)
#                              -> after every read: report (ol_confirm held18, rf_follow @18, rf_tax extract, RECEDING extract-held, th_k)
#   flow_cl.sh report [tags]   the report stage alone (after the reads)
#   flow_cl.sh body-gate <tag> | body-report [tags]   job bodies
set -uo pipefail; cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/flowhead/closed_loop; M=$DATA_DIR/runs/route_mem/read; P=$DATA_DIR/third_party/alpasim/.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python
ME=experiments/flowhead/scripts/flow_cl.sh; RF=experiments/route_follow/scripts; PRIO=${PRIO:-12}
TAGS0="FMH-F-s0 RGH-F-s0 FMH-F-s1 RGH-F-s1"; REFS="RSG-L18-s0 RSG-L18-s1"
mkdir -p $O/pool $O/check
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner flowhead --name "$name" --tries 1 --priority $PRIO --no-check \
     --log-dir $O/pool/$name "$@" | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
case $1 in
submit)
  shift; rr=()
  for T in ${*:-$TAGS0}; do
    [[ ! -e $M/$T ]] || { echo "$M/$T exists"; exit 1; }
    c=$(sub fcl-check-$T --vram 10 --cpu 6 --ram 30 --timeout-h 1 -- $TR experiments/alpasim/scripts/sh30_check.py --tag $T --n 300 --out $O/check/$T) || exit 1
    g=$(sub fcl-gate-$T --after $c --vram 0.5 --cpu 1 --ram 4 --timeout-h 1 -- bash $ME body-gate $T) || exit 1
    for i in 0 1 2 3 4 5; do
      rr+=($(sub fcl-read-$T-$i --after $g --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
             -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1
    done
  done
  printf '%s\n' "${rr[@]}" >> $O/reads.txt
  sub fcl-report --after "$(IFS=,; echo "${rr[*]}")" --vram 0.5 --cpu 16 --ram 64 --timeout-h 3 -- bash $ME body-report ${*:-$TAGS0} > /dev/null || exit 1
  tail -n 4 $O/jobs.txt ;;
report)
  shift; sub fcl-report --vram 0.5 --cpu 16 --ram 64 --timeout-h 3 -- bash $ME body-report ${*:-$TAGS0} ;;
body-gate)  # G-eq: mean of per-token max |served - bench| <= 0.05 m and p99 <= 0.15 m
  python3 - "$O/check/$2/check.json" "$O/check/$2/gate.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["summary"]["parity"]["pred_max"]
ok = s["mean"] <= 0.05 and s["p99"] <= 0.15
json.dump(dict(pred_max=s, passed=ok), open(sys.argv[2], "w"), indent=1)
print("G-eq", "pass" if ok else "FAIL", s)
sys.exit(0 if ok else 1)
PY
  ;;
body-report)
  shift; mkdir -p $O/report $O/tax; H=(); F=(); S=(); ALL=""
  for t in "$@" $REFS; do ls $M/$t/c*/run/aggregate/results-summary.json > /dev/null 2>&1 && ALL="$ALL $t"; done
  for t in $ALL; do d=$(ls -d $M/$t/c*/run | paste -sd,); H+=("$t@held18=$d"); F+=("$t@18=$d"); done
  MN=("FMH-F=FMH-F-s0,FMH-F-s1" "RGH-F=RGH-F-s0,RGH-F-s1" "RSG-L18=RSG-L18-s0,RSG-L18-s1")
  PR=("FMH-F:RGH-F" "FMH-F-s0:RGH-F-s0" "FMH-F-s1:RGH-F-s1" "FMH-F:RSG-L18" "RGH-F:RSG-L18")
  $P experiments/onpol_long/scripts/ol_confirm.py --runs "${H[@]}" --base RGH-F-s0 --mean "${MN[@]}" --pairs "${PR[@]}" --out $O/report/held18 > /dev/null; rc1=$?
  $P $RF/rf_follow.py --runs "${F[@]}" --out $O/report/follow --workers 16 --pairs FMH-F-s0@18:RGH-F-s0@18 FMH-F-s1@18:RGH-F-s1@18 > /dev/null; rc2=$?
  for t in "$@"; do
    [[ -f $O/tax/held_$t.pkl ]] || $P $RF/rf_tax.py extract --runs $(ls -d $M/$t/c*/run) --out $O/tax/held_$t.pkl --workers 16 > $O/tax/extract_$t.log 2>&1
    S+=("$t=$O/tax/held_$t.pkl")
  done
  .venv/bin/python $RF/rf_receding.py extract-held --set "${S[@]}" --follow $O/report/follow/scenes.csv --out $O/held.pkl; rc3=$?
  .venv/bin/python $RF/rf_modeavg.py thk --runs $(for t in "$@"; do [[ $t == FMH* ]] && echo "$t=$(ls -d $M/$t/c*/run | paste -sd,)"; done) --held $O/held.pkl --out $O/thk.pkl; rc4=$?
  echo "held18 $rc1 follow $rc2 receding $rc3 thk $rc4"; exit $((rc1 + rc2 + rc3 + rc4)) ;;
*) sed -n '2,8p' "$0"; exit 2 ;;
esac
