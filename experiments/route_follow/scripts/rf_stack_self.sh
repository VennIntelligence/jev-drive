#!/usr/bin/env bash
# STACK reflex module: offline self-state rows and the serving checks (rf_stack_self.py; plans/2026-10-16-stack-prereg.md 4.5).
# Box, repo root; every job through the GPU pool, ids in $O/jobs.txt.
#   rf_stack_self.sh submit <old rev>     feat (CPU) | ckpt -> offline replays + cmp | closed-loop smoke reads (3 held scenes, SH30_COMPILE=0:
#                                         RSG-L18-s0, STKRX0-test, STKRX1-test, STKRX1-test with JEV_REFLEX_OFF=1) -> smoke cmp
#   rf_stack_self.sh pai <scene>          (after ckpt: pai_box.sh checks the checkpoint at submit) 1 PAI scene of STKRX1-test -> its rx check
#   rf_stack_self.sh replays <old rev>              job body: the offline Core.plan replays (old code = a worktree at <old rev>) + cmp
# Env: PRIO (pool priority, 8).
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}
O=$DATA_DIR/runs/route_follow/stack/self; PY=experiments/route_follow/scripts/rf_stack_self.py; TR=$DATA_DIR/envs/op-train/bin/python; V=.venv/bin/python
mkdir -p $O/pool
sub() { local name=$1 after=$2; shift 2; local id; id=$($V -m jevdrive.cl submit --owner stack-reflex --name "$name" --tries 1 --priority ${PRIO:-8} --no-check \
     ${after:+--after $after} --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }
     echo "$id $name" >> $O/jobs.txt; echo "$id"; }
case ${1:?stage} in
submit)
  REV=${2:?old rev}
  sub rx-feat "" --vram 0.5 --cpu 16 --ram 40 --timeout-h 1 -- $V $PY feat --workers 16 > /dev/null || exit 1
  c=$(sub rx-ckpt "" --vram 10 --cpu 4 --ram 30 --timeout-h 1 -- $TR $PY ckpt) || exit 1
  sub rx-replay "$c" --vram 12 --cpu 8 --ram 40 --timeout-h 2 -- bash experiments/route_follow/scripts/rf_stack_self.sh replays $REV > /dev/null || exit 1
  L=$O/smoke3.txt; head -3 $DATA_DIR/runs/onpol_long/lists/held_n6/c0.txt > $L
  s=(); dirs=()
  for x in RSG-L18-s0: STKRX0-test: STKRX1-test: STKRX1-test:off; do
    T=${x%%:*}; f=${x#*:}; d=$O/smoke/$T${f:+-$f}; dirs+=($d)
    s+=($(sub rx-smoke-$T${f:+-$f} "$c" --vram 20 --cpu 6 --ram 40 --timeout-h 1 --gpus 1,2,3,4,5 --pin-strict --env CL_HIST_KEY=alpasim-read4 --env SH30_COMPILE=0 \
          ${f:+--env JEV_REFLEX_OFF=1} -- bash experiments/route_mem/scripts/rm_chain.sh readlist $T $L $d)) || exit 1
  done
  sub rx-smoke-cmp "$(IFS=,; echo "${s[*]}")" --vram 0.5 --cpu 2 --ram 8 --timeout-h 1 -- $V $PY smoke --runs "${dirs[@]}" --base RSG-L18-s0 > /dev/null || exit 1
  cat $O/jobs.txt ;;
pai)
  echo "${2:?scene}" > $O/pai1.txt
  p=$(CHUNK=1 PRIO=${PRIO:-8} bash experiments/alpasim/scripts/pai_box.sh submit $O/pai_smoke STKRX1-test $O/pai1.txt | tail -1 | awk '{print $1}')
  [[ -n $p ]] || { echo "PAI submit failed" >&2; exit 1; }; echo "$p pai-smoke" >> $O/jobs.txt
  sub rx-pai-cmp "$p" --vram 0.5 --cpu 2 --ram 8 --timeout-h 1 -- $V $PY smoke --runs $O/pai_smoke/runs/stkrx1-test_c000/sim --base stkrx1-test_c000 --suffix _pai > /dev/null || exit 1
  tail -2 $O/jobs.txt ;;
replays)
  REV=${2:?old rev}; W=$O/code_old; R=$O/replay
  [[ -d $W ]] || git worktree add --detach $W $REV || exit 1
  mapfile -t S < <($V - <<'PY'
import numpy as np                   # 3 held scenes of RSG-L18-s0's c0 rollouts with 18 dumped decisions
import os
M = np.load(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs") + "/runs/onpol_nuplan/rows/L18RSGallheld/meta.npz", allow_pickle=True)
run = sorted(set(M["run"].tolist()))[0]
sc = sorted({s for s, r in zip(M["scene"].tolist(), M["run"].tolist()) if r == run and (M["k"][(M["scene"] == s) & (M["run"] == run)] == 17).any()})[:3]
print("\n".join(f"{run}::{s}" for s in sc))
PY
)
  (( ${#S[@]} == 3 )) || { echo "replay sessions: ${S[*]}" >&2; exit 1; }
  $TR $PY replay --code $W --tag RSG-L18-s0 --out $R/old_base.npz --sessions "${S[@]}" &&
  $TR $PY replay --code . --tag RSG-L18-s0 --out $R/new_base.npz --sessions "${S[@]}" &&
  $TR $PY replay --code . --tag RSG-L18-s0 --out $R/new_base2.npz --sessions "${S[@]}" &&
  $TR $PY replay --code . --tag STKRX0-test --self --out $R/zero.npz --sessions "${S[@]}" &&
  $TR $PY replay --code . --tag STKRX1-test --self --out $R/live.npz --sessions "${S[@]}" &&
  JEV_REFLEX_OFF=1 $TR $PY replay --code . --tag STKRX1-test --self --out $R/off.npz --sessions "${S[@]}" &&
  $V $PY cmp --files $R/{old_base,new_base,new_base2,zero,live,off}.npz ;;
*) sed -n '2,8p' "$0"; exit 2 ;;
esac
