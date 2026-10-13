#!/usr/bin/env bash
# LATE-TURNS-LONG (research/route-follow/late-turns.md addendum): the 86 scenes whose first turn the @18 window does not complete, read at
# token + 19 s (num_timesteps_after counts 20 Hz steps: 380) (cache after380_lt = alpasim_long/build_cache.py --after 380 --scenes long86.txt; ~40 decisions). Box, repo root, pool jobs.
#   rf_lateturn_long.sh smoke <TAG>          3 scenes (reconstruction ends early / log ends / fully covered), one job, card 1-5
#   rf_lateturn_long.sh submit               the 86 scenes x {RSG-L18-s0, RSGboth-L18-s0, RB0-F-s0} in NCH chunks each, cards 1-5
#   rf_lateturn_long.sh scan | post          class scan of the 86 on the long window (RF_K_END=39, EV_CFG = the long run's config); rf_follow + rf_tax + shape + scores + report
#   rf_lateturn_long.sh readlist <TAG> <list> <dir>    job body (rm_chain's sim with the long cache)
set -uo pipefail; cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/route_follow/lateturn_long; CK=$DATA_DIR/runs/op_parity/runs; NCH=${NCH:-3}; MODELS="RSG-L18-s0 RSGboth-L18-s0 RB0-F-s0"
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; X=experiments/route_follow/scripts
C=$DATA_DIR/datasets/alpasim_nuplan_long/after380_lt; KEY=runtime.scene_provider.trajdata
source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
LONG=("$KEY.cache_location=$C" "$KEY.dataset.extra_params.num_timesteps_after=380")
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner route-follow --name "$name" --tries 1 --priority 14 --no-check \
     --log-dir $O/pool/$name "$@" | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
case $1 in
readlist)  T=$2; L=$3; D=$4
  [[ -f $CK/$T/ckpt-final.pt ]] || { echo "no checkpoint $T" >&2; exit 1; }; [[ ! -e $D/aggregate/results-summary.json ]] || { echo "$D already ran" >&2; exit 1; }
  rm -rf "$D"; mkdir -p "$D"
  env SH30_TAG=$T SH30_DUMP=0 bash experiments/alpasim/scripts/run.sh "$D" sh30 --scene-list "$L" "${BASE[@]}" "${LONG[@]}" || exit 1
  [[ -f $D/aggregate/results-summary.json ]] || { echo "no results-summary" >&2; exit 1; } ;;
smoke)  T=${2:-RSG-L18-s0}; mkdir -p $O/pool $O/lists
  printf '%s\n' 2021.09.09.14.18.22_veh-48_00960_01115-dcf3da6b34935bc8 2021.05.25.16.37.23_veh-25_03311_03550-46aea2dcb1485353 \
    2021.09.16.19.27.01_veh-45_01749_03230-42629decf8ad5328 > $O/lists/smoke3.txt
  sub lt-long-smoke-$T --vram 20 --cpu 6 --ram 40 --timeout-h 1 --gpus 1,2,3,4,5 --pin-strict --env CL_HIST_KEY=alpasim-read4 -- \
      bash experiments/route_follow/scripts/rf_lateturn_long.sh readlist $T $O/lists/smoke3.txt $O/smoke/$T ;;
submit) mkdir -p $O/pool $O/lists
  python3 - $O/long86.txt $O/lists $NCH <<'PY'
import random, sys
ids = open(sys.argv[1]).read().split(); random.Random(0).shuffle(ids)
for i in range(int(sys.argv[3])): open(f"{sys.argv[2]}/c0{i}.txt", "w").write("\n".join(ids[i::int(sys.argv[3])]) + "\n")
PY
  for T in RSG-L18-s0 RSGboth-L18-s0 RB0-F-s0; do for i in $(seq 0 $((NCH - 1))); do
    sub lt-long-$T-$i --vram 20 --cpu 6 --ram 40 --timeout-h 3 --gpus 1,2,3,4,5 --pin-strict --env CL_HIST_KEY=alpasim-read4 -- \
        bash experiments/route_follow/scripts/rf_lateturn_long.sh readlist $T $O/lists/c0$i.txt $O/read/$T/c$i > /dev/null || exit 1
  done; done ;;
scan)  sub lt-long-scan --vram 0.5 --cpu 4 --ram 24 --timeout-h 1 -- env RF_K_END=39 EV_CFG=$O/long_cfg.yaml $P $X/rf_lateturn.py scan \
      --held experiments/onpol_nuplan/results/split/held.txt --only $O/long86.txt --out $O/scan.csv --workers 4 ;;
submit-post)  sub lt-long-post --vram 0.5 --cpu 24 --ram 96 --timeout-h 2 -- bash experiments/route_follow/scripts/rf_lateturn_long.sh post > /dev/null ;;
post)  export RF_K_END=39 EV_CFG=$O/long_cfg.yaml
  F=(); SH=(); S=(); rc=0
  for t in $MODELS; do d=$(ls -d $O/read/$t/c* | paste -sd,); F+=("$t@long=$d"); S+=("$t=$d"); SH+=("$t=$O/tax_$t.pkl"); done
  $P $X/rf_follow.py --runs "${F[@]}" --out $O/follow --workers 24 || rc=1
  for t in $MODELS; do [[ -s $O/tax_$t.pkl ]] || $P $X/rf_tax.py extract --runs $(ls -d $O/read/$t/c*) --out $O/tax_$t.pkl --workers 5 > $O/extract_$t.log 2>&1 & done; wait
  $P $X/rf_lateturn.py shape --set "${SH[@]}" --scan $O/scan.csv --out $O/shape.csv || rc=1
  $P $X/rf_lateturn.py scores --runs "${S[@]}" --out $O/scores.json || rc=1
  $P $X/rf_lateturn.py report --scan $O/scan.csv --follow $O/follow --scores $O/scores.json --shape $O/shape.csv --out $O/report || rc=1
  exit $rc ;;
*) echo "usage: smoke|submit|readlist" >&2; exit 2 ;;
esac
