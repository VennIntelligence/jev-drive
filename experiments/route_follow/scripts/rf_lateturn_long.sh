#!/usr/bin/env bash
# LATE-TURNS-LONG (research/route-follow/late-turns.md addendum): the 86 scenes whose first turn the @18 window does not complete, read at
# token + 19 s (cache after190_lt = alpasim_long/build_cache.py --after 190 --scenes long86.txt; ~40 decisions). Box, repo root, pool jobs.
#   rf_lateturn_long.sh smoke <TAG>          3 scenes (reconstruction ends early / log ends / fully covered), one job, card 1-5
#   rf_lateturn_long.sh submit               the 86 scenes x {RSG-L18-s0, RSGboth-L18-s0, RB0-F-s0} in NCH chunks each, cards 1-5
#   rf_lateturn_long.sh readlist <TAG> <list> <dir>    job body (rm_chain's sim with the long cache)
set -uo pipefail; cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/route_follow/lateturn_long; CK=$DATA_DIR/runs/op_parity/runs; NCH=${NCH:-3}
C=$DATA_DIR/datasets/alpasim_nuplan_long/after190_lt; KEY=runtime.scene_provider.trajdata
source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
LONG=("$KEY.cache_location=$C" "$KEY.dataset.extra_params.num_timesteps_after=190")
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
*) echo "usage: smoke|submit|readlist" >&2; exit 2 ;;
esac
