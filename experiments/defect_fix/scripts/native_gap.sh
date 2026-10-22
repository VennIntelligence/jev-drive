#!/usr/bin/env bash
# NATIVE-GAP (results/native_gap.md): what separates the native arena held reads from ~0.93. Box, repo root; every job through the pool
# (priority PRIO, default 12), ids in $O/jobs.txt. No training.
#   native_gap.sh extract            CPU job: native_gap.py extract (stored native reads -> $O/{dec,win,steps,set_census}.parquet, hum.pkl)
#   native_gap.sh cf [extract id]    the counterfactual chain: the window lists (native_gap.py cflist), one dump read per (tag, shard)
#                                    (jt.sh aread's native command + SH30_DUMP), then native_gap.py cf per tag (GPU)
#   native_gap.sh dump <tag> <shard> job body: the native closed loop of <tag> on $O/lists/s<shard>.txt with every decision dumped
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/defect_fix/native_gap; ME=experiments/defect_fix/scripts/native_gap.sh; PY=experiments/defect_fix/scripts/native_gap.py
TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
AC=$DATA_DIR/datasets/alpasim_nuplan_arena/v1; KEY=runtime.scene_provider.trajdata; CFTAGS=${CFTAGS:-RSG JTN1}
declare -A CK=([RSG]=RSG-L18-s0 [JTN1]=JTN-L18-s1 [JTN0]=JTN-L18-s0 [NTV0]=NTV0-L18-s0 [B2]=STANDFMXH-L18-s1)
mkdir -p $O/pool
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner defect-fix --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-12} \
     --no-check --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
case ${1:-} in
extract)
  sub ng-extract --vram 0.1 --cpu 32 --ram 96 --timeout-h 2 -- $AP $PY extract --workers 32 ;;
cf)
  A=${2:-}; $AP $PY cflist || die "cflist"; ids=()
  for T in $CFTAGS; do d=()
    for f in $O/lists/s*.txt; do k=${f##*/s}; k=${k%.txt}
      d+=($(TRIES=2 sub ng-dump-$T-s$k --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 -- bash $ME dump $T $k)) || exit 1; done
    ids+=($(sub ng-cf-$T --after $(IFS=,; echo "${d[*]}${A:+,$A}") --vram 16 --cpu 4 --ram 32 --timeout-h 2 -- $TR $PY cf --tag $T)) || exit 1; done
  echo "$(date '+%F %T') cf chain: ${ids[*]}" | tee -a $O/stages.txt ;;
dump)
  T=$2; k=$3; D=$O/dump/$T/c$k/run
  [[ -f $D/aggregate/results-summary.json ]] && { echo "$D already read"; exit 0; }
  source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
  rm -rf $D; mkdir -p $D
  env -u JEV_HOLD SH30_TAG=${CK[$T]} SH30_DUMP=1000000 bash experiments/alpasim/scripts/run.sh $D sh30hold --scene-list $O/lists/s$k.txt "${BASE[@]}" \
    "$KEY.cache_location=$AC/s$k" || die "run.sh rc $? ($D)" ;;
*) sed -n 2,7p $0; exit 2 ;;
esac
