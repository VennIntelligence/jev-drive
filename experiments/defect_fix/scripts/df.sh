#!/usr/bin/env bash
# NR1 chain (plans/2026-10-17-nr1-prereg.md). Box, repo root; every job through the GPU pool, --after chained; ids in $O/jobs.txt.
# NR1 trainings leave JEV_RM_EAGER out (rm_train compiles, the default for new lanes); only the flags-off identity runs are eager.
#   df.sh data              (CPU) trainval reasons -> pool on1_NR1TV -> the three base draw-weight files; merged L18 pool on1_L18NR1
#   df.sh train <arm> <seed> <steps> <tag> [init]   job body; arms below
# Env: PRIO (pool priority, default 9)
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}
O=$DATA_DIR/runs/defect_fix; CK=$DATA_DIR/runs/op_parity/runs; CR=$DATA_DIR/runs/op_parity/cache; ME=experiments/defect_fix/scripts/df.sh
PY=experiments/defect_fix/scripts/df.py; TR=$DATA_DIR/envs/op-train/bin/python; NS=$DATA_DIR/envs/navsim2/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
RT=experiments/route_mem/scripts/rm_train.py
mkdir -p $O/pool
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner defect-fix --name "$name" --tries 1 --priority ${PRIO:-9} --no-check \
     --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
jid() { awk -v n="$1" '$2 == n {i = $1} END {if (i) print i; else exit 1}' $O/jobs.txt; }
fresh() { for T in "$@"; do [[ ! -e $CK/$T/ckpt-final.pt ]] || { echo "$T exists" >&2; exit 1; }; done; }
case ${1:-} in
data)
  [[ ! -e $CR/on1_NR1TV/tab.npz && ! -e $CR/on1_L18NR1/tab.npz ]] || { echo "a pool exists" >&2; exit 1; }
  r=$(sub df-reasons --vram 0.1 --cpu 48 --ram 64 --timeout-h 2 -- $NS $PY reasons --workers 48) || exit 1
  p=$(sub df-pack --after $r --vram 0.1 --cpu 4 --ram 64 --timeout-h 2 -- $TR $PY pack) || exit 1
  w=$(sub df-weights --after $p --vram 0.1 --cpu 4 --ram 48 --timeout-h 1 -- bash -c "$TR $PY weights --wait 0.05 --launch 0.05 --out D05 && \
      $TR $PY weights --wait 0.10 --launch 0.10 --out D10 && $TR $PY weights --rule all --out DALL") || exit 1
  m=$(sub df-merge-L18NR1 --vram 0.1 --cpu 4 --ram 48 --timeout-h 1 -- $TR experiments/route_follow/scripts/rf_pait.py merge \
      --pools on1_L18RB0:rb0 on1_L18RTB1:rtb1 --out L18NR1) || exit 1
  echo "data: reasons $r pack $p weights $w merge $m" | tee -a $O/stages.txt ;;
*) sed -n 2,6p $0; exit 2 ;;
esac
