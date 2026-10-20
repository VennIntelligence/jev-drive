#!/usr/bin/env bash
# Zero-copy pool merge check (jevdrive.tokens): on1_JTPHZ = on1_JTPH's merge (rf_pait.py merge, same sources and order) without copying the
# tokens, checked bit for bit against on1_JTPH, then removed. Box, repo root; every job through the GPU pool at low priority; ids in $O/jobs.txt.
#   zc_merge.sh launch    merge on1_JTPHZ (+ r2.py fix, as ROUND2 did) -> rows check (CPU) + pinned check (GPU) + R2J's command 60 steps eager
#                         on on1_JTPH and on on1_JTPHZ -> df.py bitcmp
#   zc_merge.sh wait      blocks until every job of this lane is done or one failed (one run_in_background waiter)
#   zc_merge.sh clean     removes every file this lane created (the test pool, its feat file, fix json, the two 60-step runs)
#   job bodies: merge | rows | pinned | train <A|B> | bitcmp
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/cl_infra/zc; ME=experiments/cl_infra/scripts/zc_merge.sh; PY=experiments/cl_infra/scripts/zc_check.py
TR=$DATA_DIR/envs/op-train/bin/python; V=.venv/bin/python; CR=$DATA_DIR/runs/op_parity/cache; RUNS=$DATA_DIR/runs/op_parity/runs
PARTS="on1_L18RB0:rb0:self on1_L18RTB1:rtb1:self on1_L18YS:ys:self on1_L18Y:y:self on1_NR1TV:tv:both"     # = r2.sh: on1_JTPH's parts in order
Z=on1_JTPHZ; declare -A TAG=([A]=ZCA-L18-s0 [B]=ZCB-L18-s0) POOL=([A]=on1_JTPH [B]=$Z)
mkdir -p $O/pool
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner cl_infra --name "$name" --tries 1 --priority ${PRIO:-1} \
     --no-check --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
case ${1:-} in
launch)
  [[ ! -e $CR/$Z && ! -e $CR/$Z@warp ]] || die "$Z exists"
  for A in A B; do [[ ! -e $RUNS/${TAG[$A]} ]] || die "${TAG[$A]} exists"; done
  : > $O/jobs.txt
  m=$(sub zc-merge --vram 0.1 --cpu 4 --ram 40 --timeout-h 1 -- bash $ME merge) || exit 1
  r=$(sub zc-rows --after $m --vram 0.1 --cpu 8 --ram 24 --timeout-h 2 -- bash $ME rows) || exit 1
  p=$(sub zc-pinned --after $m --vram 4 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME pinned) || exit 1
  a=$(sub zc-train-A --after $m --train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME train A) || exit 1
  b=$(sub zc-train-B --after $m --train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME train B) || exit 1
  c=$(sub zc-bitcmp --after $a,$b --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME bitcmp) || exit 1
  echo "$(date '+%F %T') launch (code $(git rev-parse --short HEAD)): merge $m rows $r pinned $p train $a $b bitcmp $c" | tee -a $O/stages.txt ;;
merge)   # = r2.sh merge with --out JTPHZ (zero-copy, the new default), then r2.py fix as ROUND2 ran it
  free=$(df -BG --output=avail $DATA_DIR | tail -1 | tr -dc 0-9); (( free >= 80 )) || die "free $free GB < 80 GB"
  $TR experiments/route_follow/scripts/rf_pait.py merge --pools $PARTS on1_PAIH:ph:both on1_PAIH05:ph05:both on1_PAIH15:ph15:both \
      --out JTPHZ --nan-keys fut_reg | tee $O/merge.json || die "merge"
  $TR experiments/round2/scripts/r2.py fix --pool $Z || die "fix"
  du -sb --apparent-size $CR/$Z $CR/$Z@warp $DATA_DIR/runs/route_mem/feat/$Z.npz | tee $O/du.txt ;;
rows)    $TR $PY rows --out $O ;;
pinned)  $TR $PY pinned --out $O ;;
train)   # R2J-L18-s0's command (runs/round2/stages.txt) with tag, pool and --steps 60
  A=$2; exec $TR experiments/route_mem/scripts/rm_train.py train --tag ${TAG[$A]} --init NTV0-L18-s0 --allow-init --warmup 1 --eval-every 500 \
    --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-split alpasim_nuplan/onpol-train --lr-scale 0.1 \
    --on-k 13 --on-mode self --on ${POOL[$A]} --on-k 64 --on-mode mixed --on-lon-all --on-normal --on-wfile runs/defect_fix/w/R2JF.npz \
    --on-hold-pct 15 --ego-scale-speed 0.5 --ego-scale-max 0.2 --lat-onc 0.5 --lat-soft 0.1 --lat-side 0.5 \
    --lat-file $DATA_DIR/runs/defect_fix/lat/L18NR1v2 --lat-depth 0.0 --lat-soft-depth 0.0 --lat-buf 0.4 --lat-host --on-arc 0 --diag \
    --warmup 500 --lr-floor 0.1 --on-arc-rows brake --steps 60 --seed 0 --flat-brk $DATA_DIR/runs/defect_fix/simp/flat_onset.txt --flat-p 0.5 ;;
bitcmp)
  $TR experiments/defect_fix/scripts/df.py bitcmp --a ${TAG[A]} --b ${TAG[B]} --out zc_identity \
      --note "R2J's command 60 steps eager: on1_JTPH (copied tokens) vs on1_JTPHZ (zero-copy front.parts.json)"; rc=$?
  mv experiments/defect_fix/results/zc_identity.json $O/identity.json; rm -f $DATA_DIR/runs/defect_fix/zc_identity.json; exit $rc ;;
wait)
  while :; do
    n=0; f=0
    while read -r id name; do [[ -f $O/pool/$name/DONE ]] && n=$((n + 1)); [[ -f $O/pool/$name/ERROR ]] && f=1; done < $O/jobs.txt
    (( f )) && { echo "a job failed"; exit 1; }; (( n == $(wc -l < $O/jobs.txt) )) && { echo "all done"; exit 0; }
    sleep 60
  done ;;
clean)
  rm -rf $CR/$Z $CR/$Z@warp $DATA_DIR/runs/route_mem/feat/$Z.npz $DATA_DIR/runs/round2/fix_$Z.json
  for A in A B; do rm -rf $RUNS/${TAG[$A]} $DATA_DIR/runs/route_mem/train-${TAG[$A]}; done ;;
*) sed -n 2,9p "$0"; exit 1 ;;
esac
