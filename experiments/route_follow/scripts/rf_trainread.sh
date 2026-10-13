#!/usr/bin/env bash
# TRAINREAD chain (rf_trainread.py docstring): closed-loop reads @18 of RSG-L18-s0, RSGboth-L18-s0, RB0-F-s0 on the pick list
# ($O/pick/scenes.txt, made by `rf_trainread.py pick`), same serving as the held reads (rm_chain.sh readlist), then rf_follow + rf_inside + the report.
# Box, repo root.   rf_trainread.sh submit | report
set -uo pipefail; cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/route_follow/trainread; M=$DATA_DIR/runs/route_mem/read; L=$DATA_DIR/runs/onpol_long/lists/held_n6
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; X=experiments/route_follow/scripts
MODELS="RSG-L18-s0 RSGboth-L18-s0 RB0-F-s0"; NCH=3
case $1 in
submit)
  mkdir -p $O/pool $O/lists; python3 - $O/pick/scenes.txt $O/lists $NCH <<'PY'   # fixed-seed shuffle, round-robin chunks (turn and control scenes balanced)
import random, sys
ids = open(sys.argv[1]).read().split(); random.Random(0).shuffle(ids)
for i in range(int(sys.argv[3])): open(f"{sys.argv[2]}/c0{i}.txt", "w").write("\n".join(ids[i::int(sys.argv[3])]) + "\n")
PY
  cat $L/c*.txt > $O/held_ids.txt
  sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner route-follow --name "$name" --tries 1 --priority 14 --no-check \
       --log-dir $O/pool/$name "$@" | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
  dep=()
  s=$(sub tr-screen-held --vram 0.5 --cpu 24 --ram 60 --timeout-h 1 -- $P $X/rf_trainread.py screen --list $O/held_ids.txt --out $O/held_class.csv --workers 24) || exit 1
  dep+=($s)
  for T in $MODELS; do for i in $(seq 0 $((NCH - 1))); do
    dep+=($(sub tr-read-$T-$i --vram 20 --cpu 6 --ram 40 --timeout-h 2 --gpus 1,2,3,4,5 --pin-strict --env CL_HIST_KEY=alpasim-read4 -- \
            bash experiments/route_mem/scripts/rm_chain.sh readlist $T $O/lists/c0$i.txt $O/read/$T/c$i)) || exit 1
  done; done
  sub tr-report --after "$(IFS=,; echo "${dep[*]}")" --vram 0.5 --cpu 32 --ram 64 --timeout-h 1 -- bash $X/rf_trainread.sh report > /dev/null || exit 1
  cat $O/jobs.txt ;;
report)
  F=(); TD=(); HD=()
  for t in $MODELS; do
    d=$(ls -d $O/read/$t/c*/run | paste -sd,); h=$(ls -d $M/$t/c*/run | paste -sd,)
    F+=("$t@train=$d" "$t@held=$h"); TD+=("$t=$d"); HD+=("$t=$h")
  done
  $P $X/rf_follow.py --runs "${F[@]}" --out $O/follow --workers 48 --pairs RSGboth-L18-s0@train:RSG-L18-s0@train RSG-L18-s0@train:RB0-F-s0@train \
     RSGboth-L18-s0@train:RB0-F-s0@train; rc1=$?
  $P $X/rf_inside.py --follow $O/follow --pairs RSGboth-L18-s0@train:RSG-L18-s0@train RSG-L18-s0@train:RB0-F-s0@train RSGboth-L18-s0@train:RB0-F-s0@train \
     --out $O/inside.json; rc2=$?
  $P $X/rf_trainread.py report --follow $O/follow --train-dirs "${TD[@]}" --held-dirs "${HD[@]}" --class-csv $O/pick/prescreen.csv $O/held_class.csv \
     --scenes $O/pick/scenes.csv --pairs RSGboth-L18-s0:RSG-L18-s0 RSG-L18-s0:RB0-F-s0 RSGboth-L18-s0:RB0-F-s0 --out $O/report; rc3=$?
  echo "follow $rc1 inside $rc2 report $rc3"; exit $((rc1 + rc2 + rc3)) ;;
*) echo "rf_trainread.sh submit | report" >&2; exit 2 ;;
esac
