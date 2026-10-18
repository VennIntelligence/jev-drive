#!/usr/bin/env bash
# JTN-DIAG (jtn_diag.py): camera reruns of JTN-L18-s0's zero / low-score windows and RSG-L18-s0 on the same windows, same serve. Box, repo root.
#   jtn_diag.sh select [args]          jtn_diag.py select (CPU, by hand, ~1 min) -> $O/sel.json + $O/lists/<src>_s<shard>_<n>.txt
#   jtn_diag.sh submit [list names]    one pool job per (list, checkpoint) -> $O/rerun/<list>_<tag>/; one collect job --after all of them.
#                                      Env PRIO (12). Ids appended to $O/jobs.txt
#   jtn_diag.sh vread <list> <tag>     job body: = jt.sh aread (arena: sh30hold, JEV_HOLD stop3 / unset, the arena shard cache) or
#                                      rf_lateturn_video.sh (held: sh30, after160 cache, 18 decisions), plus eval.video.render_video=true; no prune
#   jtn_diag.sh collect                job body: jtn_diag.py collect -> $O/videos/ + $O/selection.json
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/defect_fix/jtn_diag; PY=experiments/defect_fix/scripts/jtn_diag.py; ME=experiments/defect_fix/scripts/jtn_diag.sh
AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; AC=$DATA_DIR/datasets/alpasim_nuplan_arena/v1; KEY=runtime.scene_provider.trajdata
LONG=("$KEY.cache_location=$DATA_DIR/datasets/alpasim_nuplan_long/after160" "$KEY.dataset.extra_params.num_timesteps_after=160")
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
mkdir -p $O/pool
case ${1:-} in
select) shift; $AP $PY select "$@" ;;
submit)
  shift; L=("$@"); [[ ${#L[@]} -gt 0 ]] || L=($(cd $O/lists && ls *.txt | sed 's/\.txt$//'))
  ids=()
  for l in "${L[@]}"; do for t in JTN-L18-s0 RSG-L18-s0; do
    id=$(.venv/bin/python -m jevdrive.cl submit --owner defect-fix --name jd-$l-${t%%-*} --tries 2 --priority ${PRIO:-12} --no-check \
         --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 --log-dir $O/pool/$l-$t -- bash $ME vread $l $t < /dev/null | awk '{print $NF}')
    [[ -n $id ]] || die "submit $l $t"; ids+=($id); echo "$id jd-$l-$t" >> $O/jobs.txt; done; done
  c=$(.venv/bin/python -m jevdrive.cl submit --owner defect-fix --name jd-collect --tries 1 --priority ${PRIO:-12} --no-check --after $(IFS=,; echo "${ids[*]}") \
      --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 --log-dir $O/pool/collect -- bash $ME collect < /dev/null | awk '{print $NF}')
  echo "$c jd-collect" >> $O/jobs.txt; echo "$(date '+%F %T') ${#ids[@]} rerun jobs ${ids[0]} .. ${ids[-1]}, collect $c" | tee -a $O/stages.txt ;;
vread)
  l=$2; T=$3; D=$O/rerun/${l}_$T; LS=$O/lists/$l.txt; src=${l%%_*}
  source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
  rm -rf $D; mkdir -p $D; rc=0
  case $src in
    held) env -u JEV_HOLD SH30_TAG=$T SH30_DUMP=0 bash experiments/alpasim/scripts/run.sh $D sh30 --scene-list $LS "${BASE[@]}" "${LONG[@]}" \
            eval.video.render_video=true || rc=$? ;;
    arena|off)
      k=${l#*_s}; k=${k%%_*}; HE=(JEV_HOLD=stop3); [[ $src == off ]] && HE=(-u JEV_HOLD)
      env "${HE[@]}" SH30_TAG=$T bash experiments/alpasim/scripts/run.sh $D sh30hold --scene-list $LS "${BASE[@]}" \
        "$KEY.cache_location=$AC/s$k" eval.video.render_video=true || rc=$? ;;
    *) die "unknown source $src" ;;
  esac
  if [[ $rc -ne 0 ]]; then
    grep -q "exit code -9" "$D/native_summary.json" 2>/dev/null && { echo "runtime worker killed (-9): exit 137 for the pool's kill retry" >&2; exit 137; }
    die "run.sh rc $rc ($D)"; fi
  echo "$(find $D/rollouts -name '*.mp4' | wc -l) mp4" ;;
collect) $AP $PY collect ;;
*) die "usage: jtn_diag.sh select | submit | vread | collect" ;;
esac
