#!/usr/bin/env bash
# LEADSTART-PROBE chain (plans/2026-10-19-leadstart-probe-prereg.md). Box, repo root; every job through the GPU pool (priority PRIO, default 11),
# ids in $O/jobs.txt. No training.
#   leadstart_probe.sh submit        one --after chain: select (CPU) -> dumps (wait set v2 variant a lead windows with JTN-L18-s0, wait_set.sh crun
#                                    + SH30_DUMP; arena held lead windows with JTN-L18-s1 native, redline_probe's command) | tvsel -> tvindex -> pp_prep
#                                    (warp) -> feats -> probe -> cleanup (dump npz deleted, features kept)
#   leadstart_probe.sh dump-wsa | dump-ar <list> | cleanup     job bodies
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/defect_fix/leadstart_probe; ME=experiments/defect_fix/scripts/leadstart_probe.sh; PY=experiments/defect_fix/scripts/leadstart_probe.py
TR=$DATA_DIR/envs/op-train/bin/python; NS=$DATA_DIR/envs/navsim2/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
AC=$DATA_DIR/datasets/alpasim_nuplan_arena/v1; KEY=runtime.scene_provider.trajdata
mkdir -p $O/pool
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner defect-fix --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-11} \
     --no-check --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
case ${1:-} in
submit)
  $AP $PY select || die "select"; d=()
  d+=($(TRIES=2 sub ls-dump-wsa --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 -- bash $ME dump-wsa)) || exit 1
  for f in $O/lists/ar_*.txt; do L=$(basename $f .txt)
    d+=($(TRIES=2 sub ls-dump-$L --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 -- bash $ME dump-ar $L)) || exit 1; done
  s=$(sub ls-tvsel --vram 0.1 --cpu 32 --ram 64 --timeout-h 2 -- $TR $PY tvsel --workers 32) || exit 1
  i=$(sub ls-tvindex --after $s --vram 0.1 --cpu 4 --ram 48 --timeout-h 2 -- $NS $PY tvindex) || exit 1
  p=$(sub ls-tvprep --after $i --vram 6 --cpu 16 --ram 40 --timeout-h 3 -- $TR experiments/op_parity/scripts/pp_prep.py --data idx_leadstart_tv \
      --frames warp --no-side --synth cpu --workers 16) || exit 1
  f=$(sub ls-feats --after $(IFS=,; echo "${d[*]},$p") --vram 24 --cpu 8 --ram 64 --timeout-h 3 -- $TR $PY feats) || exit 1
  q=$(sub ls-probe --after $f --vram 24 --cpu 16 --ram 96 --timeout-h 4 -- $TR $PY probe) || exit 1
  c=$(sub ls-cleanup --after $q --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $ME cleanup) || exit 1
  echo "$(date '+%F %T') chain: dumps ${d[*]} tvsel $s tvindex $i tvprep $p feats $f probe $q cleanup $c" | tee -a $O/stages.txt ;;
dump-wsa)   # wait set v2 variant a lead-class windows, JTN-L18-s0 (launch_diag's dump of variant b: the same driver path)
  D=$O/dump/wsa; [[ -f $D/run/done ]] && { echo "$D already read"; exit 0; }
  rm -rf $D; mkdir -p $D
  SH30_DUMP=1000000 bash experiments/alpasim_proxy/scripts/wait_set.sh crun JTN-L18-s0 $D/run $O/lists/wsa.txt || die "crun rc $?" ;;
dump-ar)    # arena held windows of one list, JTN-L18-s1 native (redline_probe.sh dump's command)
  L=$2; k=${L#ar_s}; k=${k%%_*}; D=$O/dump/$L/run
  [[ -f $D/aggregate/results-summary.json ]] && { echo "$D already read"; exit 0; }
  source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
  rm -rf $D; mkdir -p $D
  env -u JEV_HOLD SH30_TAG=JTN-L18-s1 SH30_DUMP=1000000 bash experiments/alpasim/scripts/run.sh $D sh30hold --scene-list $O/lists/$L.txt "${BASE[@]}" \
    "$KEY.cache_location=$AC/s$k" || die "run.sh rc $? ($D)" ;;
cleanup)    # the dumped decision npz (frames, ~0.7 MB each) go; drive.jsonl, features and probe outputs stay
  [[ -f $O/probe/probe.json ]] || die "no probe.json"
  find $O/dump -path '*driver-logs/dump/*.npz' -delete; du -sh $O ;;
*) sed -n 2,7p $0; exit 2 ;;
esac
