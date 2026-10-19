#!/usr/bin/env bash
# LAMP-PIXELS chain (plans/2026-10-19-lamp-pixels-prereg.md). Box, repo root; every job through the GPU pool (priority PRIO, default 12),
# ids in $O/jobs.txt. No policy training. SMOKE=1 runs everything under $O/smoke with a few rows / steps.
#   lamp_pixels.sh select       pool job: lamp_pixels.py select (CPU), then read $O/lists before `submit`
#   lamp_pixels.sh submit       one --after chain: one dump read per list (JTN-L18-s1 native, SH30_DUMP) -> frames -> cnn N | cnn L | tok -> ft -> eval
#   lamp_pixels.sh dump <list>  job body: redline_probe.sh dump's command on $O/lists/<list>.txt
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR; SMOKE=${SMOKE:-0}; export SMOKE
O=$DATA_DIR/runs/defect_fix/lamp_pixels; [[ $SMOKE == 1 ]] && O=$O/smoke
ME=experiments/defect_fix/scripts/lamp_pixels.sh; PY=experiments/defect_fix/scripts/lamp_pixels.py
TR=$DATA_DIR/envs/op-train/bin/python; AC=$DATA_DIR/datasets/alpasim_nuplan_arena/v1; KEY=runtime.scene_provider.trajdata
mkdir -p $O/pool
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner defect-fix --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-12} \
     --no-check --log-dir $O/pool/$name --env SMOKE=$SMOKE "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
S=$([[ $SMOKE == 1 ]] && echo -smoke)
case ${1:-} in
select) sub lp-select$S --vram 0.1 --cpu 32 --ram 96 --timeout-h 1 -- $TR $PY select --workers 32 ;;
submit)
  d=()
  for f in $O/lists/*.txt; do [[ -f $f ]] || continue; L=$(basename $f .txt)
    d+=($(TRIES=2 sub lp-dump$S-$L --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 -- bash $ME dump $L)) || exit 1; done
  A=(); [[ ${#d[@]} -gt 0 ]] && A=(--after $(IFS=,; echo "${d[*]}"))
  fr=$(sub lp-frames$S "${A[@]}" --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- $TR $PY frames) || exit 1
  n=$(sub lp-cnnN$S --after $fr --vram 40 --cpu 16 --ram 64 --timeout-h 3 -- $TR $PY cnn --var N) || exit 1
  l=$(sub lp-cnnL$S --after $fr --vram 16 --cpu 16 --ram 64 --timeout-h 3 -- $TR $PY cnn --var L) || exit 1
  t=$(sub lp-tok$S --after $fr --vram 24 --cpu 16 --ram 96 --timeout-h 3 -- $TR $PY tok) || exit 1
  f=$(sub lp-ft$S --after $n --vram 30 --cpu 12 --ram 48 --timeout-h 2 -- $TR $PY ft --workers 12) || exit 1
  e=$(sub lp-eval$S --after $n,$l,$t,$f --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash -c "$TR $PY eval && touch $O/DONE") || exit 1
  echo "$(date '+%F %T') chain: dumps ${d[*]} frames $fr cnnN $n cnnL $l tok $t ft $f eval $e" | tee -a $O/stages.txt ;;
dump)
  L=$2; k=${L#s}; k=${k%%_*}; D=$O/dump/$L/run
  [[ -f $D/aggregate/results-summary.json ]] && { echo "$D already read"; exit 0; }
  source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
  rm -rf $D; mkdir -p $D
  env -u JEV_HOLD SH30_TAG=JTN-L18-s1 SH30_DUMP=1000000 bash experiments/alpasim/scripts/run.sh $D sh30hold --scene-list $O/lists/$L.txt "${BASE[@]}" \
    "$KEY.cache_location=$AC/s$k" || die "run.sh rc $? ($D)" ;;
*) sed -n 2,7p $0; exit 2 ;;
esac
