#!/usr/bin/env bash
# ROUND9 CB2 (clean B2; plans/2026-10-25-cb2.md = the registration): a clean nuPlan model from the shipped weights that ties B2 (STANDFMXH-L18-s1).
# Box, repo root; every job through the GPU pool; ids in $O/jobs.txt, stages in $O/stages.txt, lane dir $O = $DATA_DIR/runs/round9/cb2.
#   cb2.sh step0        offline step 0 (before registration): nuplan_dag's b3 teacher on existing nuPlan rows (RB0-F-s0 onpol-train L18RB0, D3
#                       L18D3, JTN-L18-s0 practice starts PN0-5) -> cb2.py creep (human-standing DAgger rows, rule candidates, G0) -> $O/STEP0
#   cb2.sh wait <M>     one run_in_background waiter: $O/<M> or a failed / cancelled job of this lane
#   job bodies: teach <rows> <out dir>
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round9/cb2; ME=experiments/round9/scripts/cb2.sh; PY=experiments/round9/scripts/cb2.py; RES=experiments/round9/results/cb2
V=.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; AS=$DATA_DIR/third_party/alpasim; AP=$AS/.venv/bin/python
NT=experiments/nuplan_dag/scripts/nd_teach.py; EA=(env JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD)
PRIO=${PRIO:-14}
mkdir -p $O/pool $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner round9-cb2 --name "$name" --tries ${TRIES:-1} --priority $PRIO --no-check \
     --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
case ${1:-} in
step0)
  [[ ! -e $O/STEP0 || -n ${RERUN_CREEP:-} ]] || die "step 0 already ran"; mkdir -p $O/step0/teach
  echo "$(git rev-parse --short HEAD) step0" >> $O/code.txt; P0="L18RB0 L18D3 PN0 PN1 PN2 PN3 PN4 PN5"; ids=()
  for n in $P0; do ids+=($(sub cb2-s0-teach-$n --vram 0.1 --cpu 16 --ram 64 --timeout-h 2 -- bash $ME teach $n $O/step0/teach)) || exit 1; done
  c=$(sub cb2-s0-creep --after $(csv ${ids[@]}) --vram 0.1 --cpu 2 --ram 48 --timeout-h 1 -- \
      bash -c "$V $PY creep --pools $P0 --teach-dir $O/step0/teach --out $O/step0/creep && cp $O/step0/creep.json $O/step0/creep.md $RES/ && date > $O/STEP0") || exit 1
  echo "$(date '+%F %T') step0 (code $(git rev-parse --short HEAD)): teach ${ids[*]} creep $c" | tee -a $O/stages.txt ;;
creep)  # step 0 amendment (before registration): creep re-run with rule R3 added (teacher files unchanged)
  c=$(sub cb2-s0-creep2 --vram 0.1 --cpu 2 --ram 48 --timeout-h 1 -- \
      bash -c "$V $PY creep --pools L18RB0 L18D3 PN0 PN1 PN2 PN3 PN4 PN5 --teach-dir $O/step0/teach --out $O/step0/creep2 && cp $O/step0/creep2.json $O/step0/creep2.md $RES/ && date > $O/STEP0B") || exit 1
  echo "$(date '+%F %T') creep2 (code $(git rev-parse --short HEAD)): $c" | tee -a $O/stages.txt ;;
teach)  # teach <rows> <dir>: nd_teach.py teach (b3, unedited) on runs/onpol_nuplan/rows/<rows> -> <dir>/<rows>.npz
  [[ ! -e $3/$2.npz ]] || die "$3/$2.npz exists"; "${EA[@]}" $AP $NT teach --name $2 --out $3/$2.npz --workers 16 ;;
wait)
  M=$O/${2:?marker}
  until [[ -f $M ]] || ! $V - $O/jobs.txt <<'PY'
import sys
from jevdrive.cl import pool as P
last = {}
for l in open(sys.argv[1]):
    if l.split():
        last[l.split()[1]] = l.split()[0]
jobs = P.snapshot()[0]["jobs"]
bad = [f"{n} {i} {jobs[i]['state']}: {jobs[i].get('why', '')}" for n, i in last.items() if jobs.get(i, {}).get("state") in ("failed", "cancelled")]
if bad:
    print("failed:", *bad, sep="\n  ", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
  do sleep 300; done
  [[ -f $M ]] && echo "DONE $M $(cat $M)"; tail -3 $O/stages.txt; true ;;
*) sed -n 2,8p "$0"; exit 2 ;;
esac
