#!/usr/bin/env bash
# WARP-LOOP (plans/2026-10-19-warp-loop-prereg.md, results/warp_loop.md). Box, repo root; every job through the pool (priority PRIO,
# default 12), ids in $O/jobs.txt. No training.
#   warp_loop.sh cf            stage 1: NTV0-L18-s0 native dumps of native_gap's 270 windows (native_gap.sh dump) -> warp_loop.py cf per
#                              dump run (JTN1 / RSG / NTV0 native_gap dumps, JTN1 redline_probe static dumps) -> report
#   warp_loop.sh aread <chunk> stage 2 job body: JTN-L18-s1 native arena held chunk with SH30_WARP=${WARP:-const} (jt.sh aread off's command)
#   warp_loop.sh ident         stage 2 identity: 12 chunk-0 windows, switch off (WARP=none) and on (const), against the stored native read
#   warp_loop.sh s2            stage 2: the 18 chunks -> $O/arena/JTN-L18-s1@warp-$WARP/c<chunk>/run
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
R=$DATA_DIR/runs/defect_fix; O=$R/warp_loop; NG=$R/native_gap; RP=$R/redline_probe; ME=experiments/defect_fix/scripts/warp_loop.sh
PY=experiments/defect_fix/scripts/warp_loop.py; TR=$DATA_DIR/envs/op-train/bin/python
AH=$DATA_DIR/runs/alpasim_proxy/arena/hold; AC=$DATA_DIR/datasets/alpasim_nuplan_arena/v1; KEY=runtime.scene_provider.trajdata
mkdir -p $O/pool
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner defect-fix --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-12} \
     --no-check --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
case ${1:-} in
cf)
  ids=()
  for k in 0 1 3 4 5 6; do
    d=$(TRIES=2 sub wl-dump-NTV0-s$k --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 -- bash experiments/defect_fix/scripts/native_gap.sh dump NTV0 $k) || exit 1
    ids+=($(sub wl-cf-NTV0-c$k --after $d --vram 10 --cpu 8 --ram 24 --timeout-h 2 -- $TR $PY cf --tag NTV0 --run $NG/dump/NTV0/c$k/run)) || exit 1
    for T in JTN1 RSG; do ids+=($(sub wl-cf-$T-c$k --vram 10 --cpu 8 --ram 24 --timeout-h 2 -- $TR $PY cf --tag $T --run $NG/dump/$T/c$k/run)) || exit 1; done; done
  for D in $RP/dump/*/run; do L=$(basename $(dirname $D))
    ids+=($(sub wl-cf-rp-$L --vram 10 --cpu 8 --ram 24 --timeout-h 3 -- $TR $PY cf --tag JTN1 --run $D)) || exit 1; done
  r=$(sub wl-report --after $(csv ${ids[@]}) --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- .venv/bin/python $PY report) || exit 1
  echo "$(date '+%F %T') stage 1: ${#ids[@]} cf jobs ${ids[0]} .. ${ids[-1]}, report $r" | tee -a $O/stages.txt ;;
aread)
  c=$2; W=${WARP:-const}; SW=$W; [[ $W == none ]] && SW=; D=$O/arena/JTN-L18-s1@warp-$W${LIST:+-id}/c$c/run   # WARP=none: switch off
  [[ -f $D/aggregate/results-summary.json ]] && { echo "$D already read"; exit 0; }
  k=$(python3 -c "import json; print(next(x['shard'] for x in json.load(open('$AH/meta.json'))['chunks'] if x['chunk'] == $c))") || die "no chunk $c"
  source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
  rm -rf $D; mkdir -p $D
  env -u JEV_HOLD SH30_TAG=JTN-L18-s1 SH30_WARP=$SW bash experiments/alpasim/scripts/run.sh $D sh30hold --scene-list ${LIST:-$AH/lists/c$c.txt} "${BASE[@]}" \
    "$KEY.cache_location=$AC/s$k" || { grep -q "exit code -9" "$D/native_summary.json" 2>/dev/null && exit 137; die "run.sh rc ($D)"; } ;;
s2)
  W=${WARP:-const}; ids=()
  for c in $(python3 -c "import json; print(' '.join(str(x['chunk']) for x in json.load(open('$AH/meta.json'))['chunks']))"); do
    ids+=($(PRIO=$((${PRIO:-12} + 1)) TRIES=2 sub wl-ar-$W-c$c --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 -- env WARP=$W bash $ME aread $c)) || exit 1; done
  echo "$(date '+%F %T') stage 2 ($W): ${ids[*]}" | tee -a $O/stages.txt ;;
ident)
  mkdir -p $O/ident; head -12 $AH/lists/c0.txt > $O/ident/c0.txt
  for W in none const; do sub wl-id-$W --vram 20 --cpu 6 --ram 40 --timeout-h 1 --env CL_HIST_KEY=alpasim-read4 -- env WARP=$W LIST=$O/ident/c0.txt bash $ME aread 0 || exit 1; done ;;
*) sed -n 2,8p $0; exit 2 ;;
esac
