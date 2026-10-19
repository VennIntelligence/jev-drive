#!/usr/bin/env bash
# REDLINE-PROBE chain (plans/2026-10-19-redline-probe-prereg.md). Box, repo root; every job through the GPU pool (priority PRIO, default 12),
# ids in $O/jobs.txt. No training.
#   redline_probe.sh submit           select (here) -> one dump read per list (JTN-L18-s1 native, SH30_DUMP) -> feats -> probe, one --after chain
#   redline_probe.sh dump <list>      job body: JTN-L18-s1's native arena closed loop on $O/lists/<list>.txt with every decision dumped
#                                     (native_gap.sh dump's command; the list's shard = the trajdata cache)
#   redline_probe.sh feats | probe    job bodies: redline_probe.py feats / probe
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/defect_fix/redline_probe; ME=experiments/defect_fix/scripts/redline_probe.sh; PY=experiments/defect_fix/scripts/redline_probe.py
TR=$DATA_DIR/envs/op-train/bin/python; AC=$DATA_DIR/datasets/alpasim_nuplan_arena/v1; KEY=runtime.scene_provider.trajdata
mkdir -p $O/pool
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner defect-fix --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-12} \
     --no-check --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
case ${1:-} in
submit)
  .venv/bin/python $PY select || die "select"; d=()
  for f in $O/lists/*.txt; do L=$(basename $f .txt)
    d+=($(TRIES=2 sub rp-dump-$L --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 -- bash $ME dump $L)) || exit 1; done
  f=$(sub rp-feats --after $(IFS=,; echo "${d[*]}") --vram 24 --cpu 8 --ram 64 --timeout-h 2 -- bash $ME feats) || exit 1
  p=$(sub rp-probe --after $f --vram 16 --cpu 16 --ram 64 --timeout-h 2 -- bash $ME probe) || exit 1
  echo "$(date '+%F %T') chain: dumps ${d[*]} feats $f probe $p" | tee -a $O/stages.txt ;;
dump)
  L=$2; k=${L#s}; k=${k%%_*}; D=$O/dump/$L/run
  [[ -f $D/aggregate/results-summary.json ]] && { echo "$D already read"; exit 0; }
  source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
  rm -rf $D; mkdir -p $D
  env -u JEV_HOLD SH30_TAG=JTN-L18-s1 SH30_DUMP=1000000 bash experiments/alpasim/scripts/run.sh $D sh30hold --scene-list $O/lists/$L.txt "${BASE[@]}" \
    "$KEY.cache_location=$AC/s$k" || die "run.sh rc $? ($D)" ;;
feats) $TR $PY feats ;;
probe) $TR $PY probe ;;
*) sed -n 2,7p $0; exit 2 ;;
esac
