#!/usr/bin/env bash
# LATE-TURNS chain (rf_lateturn.py docstring). Box, repo root.   rf_lateturn.sh submit | post | video <list> | report
#   submit: @18 reads of RSG-L18-s0, RSGboth-L18-s0, RB0-F-s0 on $O/pick.txt (rm_chain readlist, the held reads' serving), then `post`.
#   post:   rf_follow on the new reads; rf_tax extract of the trainread + new reads; turn shape (rf_ledger) on every class rollout; scores; report.
#   video <list>: the camera rerun of RSG / RSGboth on the review scenes (tmp/held_turns/video_run.sh = rm_chain sim + render_video), card 0.
set -uo pipefail; cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/route_follow/lateturn; TRD=$DATA_DIR/runs/route_follow/trainread; M=$DATA_DIR/runs/route_mem/read; LG=$DATA_DIR/runs/route_follow/ledger
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; X=experiments/route_follow/scripts
MODELS="RSG-L18-s0 RSGboth-L18-s0 RB0-F-s0"; NCH=3
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner route-follow --name "$name" --tries 1 --priority 14 --no-check \
     --log-dir $O/pool/$name "$@" | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
case $1 in
submit)
  mkdir -p $O/pool $O/lists; python3 - $O/pick.txt $O/lists $NCH <<'PY'
import random, sys
ids = open(sys.argv[1]).read().split(); random.Random(0).shuffle(ids)
for i in range(int(sys.argv[3])): open(f"{sys.argv[2]}/c0{i}.txt", "w").write("\n".join(ids[i::int(sys.argv[3])]) + "\n")
PY
  dep=()
  for T in $MODELS; do for i in $(seq 0 $((NCH - 1))); do
    dep+=($(sub lt-read-$T-$i --vram 20 --cpu 6 --ram 40 --timeout-h 2 --gpus 1,2,3,4,5 --pin-strict --env CL_HIST_KEY=alpasim-read4 -- \
            bash experiments/route_mem/scripts/rm_chain.sh readlist $T $O/lists/c0$i.txt $O/read/$T/c$i)) || exit 1
  done; done
  sub lt-post --after "$(IFS=,; echo "${dep[*]}")" --vram 0.5 --cpu 32 --ram 96 --timeout-h 2 -- bash $X/rf_lateturn.sh post > /dev/null || exit 1
  cat $O/jobs.txt ;;
post)
  F=(); S=(); rc=0
  for t in $MODELS; do
    d=$(ls -d $O/read/$t/c* | paste -sd,); F+=("$t@new=$d")
    S+=("$t=$(ls -d $TRD/read/$t/c* | paste -sd,)" "$t=$d" "$t=$(ls -d $M/$t/c*/run | paste -sd,)")
  done
  $P $X/rf_follow.py --runs "${F[@]}" --out $O/follow_new --workers 30 || rc=1
  for t in $MODELS; do
    [[ -s $O/tax_train_$t.pkl ]] || $P $X/rf_tax.py extract --runs $(ls -d $TRD/read/$t/c*) --out $O/tax_train_$t.pkl --workers 5 > $O/extract_train_$t.log 2>&1 &
    [[ -s $O/tax_new_$t.pkl ]] || $P $X/rf_tax.py extract --runs $(ls -d $O/read/$t/c*) --out $O/tax_new_$t.pkl --workers 5 > $O/extract_new_$t.log 2>&1 &
  done; wait
  SH=()
  for t in $MODELS; do SH+=("$t=$LG/held_$t.pkl" "$t=$O/tax_train_$t.pkl" "$t=$O/tax_new_$t.pkl"); done
  $P $X/rf_lateturn.py shape --set "${SH[@]}" --scan $O/scan.csv --out $O/shape.csv || rc=1
  $P $X/rf_lateturn.py scores --runs "${S[@]}" --out $O/scores.json || rc=1
  bash $X/rf_lateturn.sh report || rc=1
  exit $rc ;;
report)
  $P $X/rf_lateturn.py report --scan $O/scan.csv --follow $TRD/follow $O/follow_new --scores $O/scores.json --shape $O/shape.csv --out $O/report ;;
video)
  L=$2; V=$O/video; mkdir -p $V/lists; split -n r/3 -d --additional-suffix=.txt $L $V/lists/c
  for T in RSG-L18-s0 RSGboth-L18-s0; do for f in $V/lists/c*.txt; do i=$(basename $f .txt)
    sub lt-vid-$T-$i --vram 12 --cpu 6 --ram 40 --timeout-h 1 --gpus 0 --pin-strict --env CL_HIST_KEY=alpasim-read4 -- \
        bash $X/rf_lateturn_video.sh $T $f $V/$T/$i > /dev/null || exit 1
  done; done; cat $O/jobs.txt | grep lt-vid ;;
*) echo "rf_lateturn.sh submit | post | report | video <list>" >&2; exit 2 ;;
esac
