#!/usr/bin/env bash
# ROUND9 BODY-SCHEMA lane (research/body-schema/notes.md section 0): pool jobs on the box. The code may sit outside the repo (uncommitted copy):
#   bash $CODE/bsl.sh prep     prep only (CPU job)
#   bash $CODE/bsl.sh smoke    fold 0: 40 training steps + eval on 2 000 rows (GPU job), after prep
#   bash $CODE/bsl.sh all      fwd (CPU, 2 ckpts x 2 shards) ; train f0..f3 (GPU) -> eval fk (GPU, after train fk + every fwd) -> report -> $O/DONE
#   bash $CODE/bsl.sh wait     block until $O/DONE, or any lane job's log dir has ERROR (-> exit 1); run it with run_in_background
set -uo pipefail
DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
CODE=$(cd "$(dirname "$0")" && pwd); REPO=${JEV_REPO:-$DATA_DIR/jev-drive}; cd $REPO
O=$DATA_DIR/runs/round9/body_schema; mkdir -p $O/pool
TR=$DATA_DIR/envs/op-train/bin/python; PY="env JEV_REPO=$REPO $TR $CODE/bsl.py"
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner body-schema --name "$name" --tries 1 --priority ${PRIO:-2} \
     --no-check --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }
     echo "$id $name" >> $O/jobs.txt; echo "$id"; }
case ${1:-} in
prep) sub bsl-prep --vram 0.1 --cpu 2 --ram 24 --timeout-h 1 -- $PY prep ;;
smoke) sub bsl-smoke --vram 10 --cpu 6 --ram 48 --timeout-h 1 -- bash -c "$PY train --fold 0 --steps 40 --ref-steps 20 && $PY eval --fold 0 --limit 2000" ;;
all)
  rm -f $O/DONE; F=(); E=()
  for c in NTV0-L18-s0 PCL1-F-s0; do for i in 0 1; do
    F+=($(sub bsl-fwd-$c-$i --vram 0.1 --cpu 12 --ram 64 --timeout-h 4 -- $PY fwd --ckpt $c --shard $i/2 --threads 12)) || exit 1; done; done
  for k in 0 1 2 3; do
    t=$(sub bsl-train-f$k --vram 10 --cpu 6 --ram 48 --timeout-h 4 -- $PY train --fold $k) || exit 1
    E+=($(sub bsl-eval-f$k --after $t,$(IFS=,; echo "${F[*]}") --vram 10 --cpu 6 --ram 48 --timeout-h 3 -- $PY eval --fold $k)) || exit 1; done
  sub bsl-report --after $(IFS=,; echo "${E[*]}") --vram 0.1 --cpu 4 --ram 24 --timeout-h 1 -- bash -c "$PY report && date > $O/DONE" > /dev/null || exit 1
  echo "fwd ${F[*]} eval ${E[*]}" ;;
wait)
  while :; do
    [[ -f $O/DONE ]] && { echo DONE; exit 0; }
    e=$(ls $O/pool/*/ERROR 2>/dev/null | head -3); [[ -n $e ]] && { echo "ERROR: $e"; exit 1; }
    [[ -n ${2:-} && -f $O/pool/$2/DONE ]] && { echo "$2 DONE"; exit 0; }
    sleep 60; done ;;
*) sed -n 2,6p $0; exit 2 ;;
esac
