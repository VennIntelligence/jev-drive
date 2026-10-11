#!/usr/bin/env bash
# Lane ONPOL-NUPLAN: rollout clips of every fixed and broken scene of the arms against the control (s0, s1), by COL1's clip pipeline
# (on_clips.py = col1_lead.py + col1_clip.py, unedited, with their state under $ON_CLIPS). Run after on_report.py has written
# experiments/onpol_nuplan/results/read/{on_stats.json, on_per_scene.csv}:
#   scripts/tmux_run.sh on-clips bash experiments/onpol_nuplan/scripts/on_clips.sh [on_clips.py cases arguments, e.g. --arms ON-SELF ON-IMIT]
#   test on train-split scenes:  ON_CLIPS=$DATA_DIR/runs/onpol_nuplan/clips-test bash .../on_clips.sh --test-run <tag>=<run dir> --limit 3
# Steps: cases -> msgs -> maps (AlpaSim venv, CPU) -> replay (one pool job: op-train env, 8 GB) -> clip --webp -> index. ON_PRUNE=1 then deletes
# the rollout.asl of passed rollouts under read/held (only useful after ON_KEEP_ASL=all). Rerunning resumes: finished messages, replays of an
# unchanged case list and rendered GIFs are kept.
# State: $ON_CLIPS (default $DATA_DIR/runs/onpol_nuplan/clips)/{STATUS, DONE, ERROR, log.txt, cases.csv, gif/, webp/, index.md}. Copy index.md to
# experiments/onpol_nuplan/results/read/clips.md; GIFs stay on the box (<= 1.4 MiB each; the repo takes clips under ~500 KB).
set -uo pipefail
cd "$(dirname "$0")/../../.."
export ON_CLIPS=${ON_CLIPS:-$DATA_DIR/runs/onpol_nuplan/clips}
O=$ON_CLIPS; S=experiments/onpol_nuplan/scripts/on_clips.py; SRC=${ALPASIM_SRC:-$DATA_DIR/third_party/alpasim}; AP=$SRC/.venv/bin/python
mkdir -p "$O"; rm -f "$O/DONE" "$O/ERROR"
exec > >(tee -a "$O/log.txt") 2>&1
status() { echo "$(date '+%F %T') on-clips: $*" | tee "$O/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$O/ERROR"; exit 1; }
status "code $(git rev-parse --short HEAD); cases"
"$AP" $S cases "$@" || die "cases"
if (( $(wc -l < "$O/cases.csv") > 1 )); then
  status "msgs and maps of $(( $(wc -l < "$O/cases.csv") - 1 )) rollouts"
  "$AP" $S msgs || die "msgs"
  "$AP" $S maps || die "maps"
  if [[ ! -f $O/lead/REPLAY_DONE ]]; then
    P=$O/pool-$(date +%Y%m%d-%H%M%S)
    id=$(.venv/bin/python -m jevdrive.cl submit --owner alpasim-onpol --name onpol-clips-replay --vram 8 --cpu 8 --ram 24 --timeout-h 2 --priority "${OT_PRIO:-12}" --log-dir "$P" -- \
         env ALPASIM_SRC="$SRC" ON_CLIPS="$O" "$DATA_DIR/envs/op-train/bin/python" $S replay | awk 'END {print $NF}') || die "submit replay"
    status "replay: pool job $id"
    until [[ -f $P/DONE || -f $P/ERROR ]]; do sleep 20; done
    [[ -f $P/DONE && -f $O/lead/REPLAY_DONE ]] || die "replay failed (see $P/log.txt)"
  fi
  status "rendering"
  "$AP" $S clip --webp || die "clip (see log.txt; rerun renders the missing ones)"
else
  status "no fixed or broken scene has a rollout log: nothing to render"
fi
"$AP" $S index > /dev/null || die "index"
[[ -z ${ON_PRUNE:-} ]] || "$AP" $S prune || die "prune"
status "all done: $(ls "$O/gif" 2>/dev/null | wc -l) GIFs, $(du -sh "$O/gif" 2>/dev/null | cut -f1); index $O/index.md"; date '+%F %T' > "$O/DONE"
