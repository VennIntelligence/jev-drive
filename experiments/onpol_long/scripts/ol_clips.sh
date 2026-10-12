#!/usr/bin/env bash
# ONPOL-LONG zero-score clips of YS-L18-s1 (default): extract simulator state of the failed rollouts -> zeros table -> stratified sample ->
# on_clips.py msgs / maps / replay (pool job) / clip. State: $ON_CLIPS (default $DATA_DIR/runs/onpol_long/zeros/$TAG); read-only on the read run.
#   scripts/tmux_run.sh ol-clips bash experiments/onpol_long/scripts/ol_clips.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
TAG=${TAG:-YS-L18-s1}
export ON_CLIPS=${ON_CLIPS:-$DATA_DIR/runs/onpol_long/zeros/$TAG}
O=$ON_CLIPS; SRC=${ALPASIM_SRC:-$DATA_DIR/third_party/alpasim}; AP=$SRC/.venv/bin/python
OC=experiments/onpol_nuplan/scripts/on_clips.py; mkdir -p "$O"; rm -f "$O/DONE" "$O/ERROR"
exec > >(tee -a "$O/log.txt") 2>&1
die() { echo "ERROR $*" | tee "$O/ERROR"; exit 1; }
[[ -f $O/read_sim.pkl ]] || "$AP" experiments/alpasim/scripts/c1_extract.py logs --run "$DATA_DIR/runs/onpol_long/read/$TAG/run" --out "$O/read_sim.pkl" || die extract
"$AP" experiments/onpol_long/scripts/ol_clips.py table --tag "$TAG" || die table
"$AP" experiments/onpol_long/scripts/ol_clips.py cases --tag "$TAG" || die cases
"$AP" $OC msgs || die msgs
"$AP" $OC maps || die maps
if [[ ! -f $O/lead/REPLAY_DONE ]]; then
  P=$O/pool-$(date +%Y%m%d-%H%M%S)
  id=$(.venv/bin/python -m jevdrive.cl submit --owner alpasim-onpol --name ol-zeros-replay --vram 8 --cpu 8 --ram 24 --timeout-h 2 --priority 12 --log-dir "$P" -- \
       env ALPASIM_SRC="$SRC" ON_CLIPS="$O" "$DATA_DIR/envs/op-train/bin/python" $OC replay | awk 'END {print $NF}') || die submit
  until [[ -f $P/DONE || -f $P/ERROR ]]; do sleep 20; done
  [[ -f $P/DONE && -f $O/lead/REPLAY_DONE ]] || die "replay failed ($P/log.txt)"
fi
"$AP" $OC clip --webp || die clip
echo "all done: $(ls $O/gif | wc -l) GIFs"; date > "$O/DONE"
