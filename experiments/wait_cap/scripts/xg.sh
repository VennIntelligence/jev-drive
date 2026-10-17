#!/usr/bin/env bash
# EXPLORE-G chain (nx_turns.py; results/next/explore_g.md). Box, repo root; every job through the GPU pool, --after chained; ids in $X/jobs.txt.
#   xg.sh launch   moves the first replay (empty route memory) aside, then: geo (CPU) | msgs (CPU) -> replay, one GPU job per checkpoint
#                  -> olg (CPU, also after geo) | lanes (CPU); nx.py hist (dev-row plans) of the four checkpoints without one (GPU)
#   xg.sh smoke    geo --limit 2 into explore_g_smoke2 (CPU, foreground, < 1 min)
# Env: PRIO (pool priority, default 9)
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}
X=$DATA_DIR/runs/wait_cap/next; O=$X/explore_g; P=experiments/wait_cap/scripts/nx_turns.py
TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
CK=(RSG-L18-s0 RTB1O-L18-s0 RTB1ARO-L18-s0 RSG-L18-s1 RTB1OB1-L18-s1)
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner wait-next --name "$name" --tries 1 --priority ${PRIO:-9} --no-check \
     --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $X/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
case ${1:-} in
launch)
  mkdir -p $O/pool
  if [[ -d $O/replay && ! -e $O/replay_emptymem ]]; then mv $O/replay $O/replay_emptymem; mv $O/replay_identity.json $O/replay_emptymem/ 2>/dev/null; fi
  [[ ! -e $O/replay ]] || { echo "$O/replay exists" >&2; exit 1; }
  g=$(sub xg2-geo --vram 0.1 --cpu 48 --ram 96 --timeout-h 1 -- $AP $P geo --out $O --workers 48) || exit 1
  m=$(sub xg2-msgs --vram 0.1 --cpu 16 --ram 32 --timeout-h 1 -- $AP $P msgs --out $O) || exit 1
  rr=(); for c in ${CK[@]}; do rr+=($(sub xg2-replay-$c --after $m --vram 12 --cpu 8 --ram 40 --timeout-h 2 --env ALPASIM_SRC=$DATA_DIR/third_party/alpasim \
      -- $TR $P replay --out $O --ckpts $c)) || exit 1; done
  o=$(sub xg2-olg --after $g,$(csv ${rr[@]}) --vram 0.1 --cpu 32 --ram 64 --timeout-h 1 -- $AP $P olg --out $O --workers 32) || exit 1
  l=$(sub xg2-lanes --after $(csv ${rr[@]}) --vram 0.1 --cpu 32 --ram 64 --timeout-h 1 -- $AP $P lanes --out $O) || exit 1
  h=$(sub xg2-hist --vram 24 --cpu 6 --ram 80 --timeout-h 1 -- $TR experiments/wait_cap/scripts/nx.py hist --tags RTB1O-L18-s0 RTB1ARO-L18-s0 RTB1OB1-L18-s1 RSG-L18-s1) || exit 1
  echo "xg launch: geo $g msgs $m replay $(csv ${rr[@]}) olg $o lanes $l hist $h" | tee -a $X/stages.txt ;;
smoke)
  rm -rf $X/explore_g_smoke2; $AP $P geo --out $X/explore_g_smoke2 --limit 2 --workers 8 ;;
*) sed -n 2,6p "$0"; exit 1 ;;
esac
