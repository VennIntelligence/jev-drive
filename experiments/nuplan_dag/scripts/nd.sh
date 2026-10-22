#!/usr/bin/env bash
# NUPLAN-DAG (plans/2026-10-22-nuplan-dag.md): the PAI DAgger map-teacher method moved to nuPlan (AlpaSim nuPlan closed loop).
# Part 1: PAI checkpoints (R8K-L18-s2, R6DAG-L18-s1) on nuPlan held 460 @18 and the arena held split (native serve, B2's adoption reads),
# paired against the stored B2 (STANDFMXH-L18-s1) and NTV0-L18-s0 reads. Box, repo root; every job through the GPU pool at PRIO 22;
# ids in $O/jobs.txt, stages in $O/stages.txt.
#   nd.sh p1 [tags]       arena 18 chunks (jt.sh aread <tag> off) + held 460 (rm_chain.sh read <tag> i 6) per tag -> p1sum
#   nd.sh wait <marker>   blocks until $O/<marker>, $O/VERDICT or a failed / cancelled job of this lane (one run_in_background waiter)
#   job bodies: p1sum
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=${ND_O:-$DATA_DIR/runs/nuplan_dag}; ME=experiments/nuplan_dag/scripts/nd.sh; PY=experiments/nuplan_dag/scripts/nd.py; V=.venv/bin/python
TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; RES=experiments/nuplan_dag/results
JT=experiments/defect_fix/scripts/jt.sh; FM=experiments/defect_fix/scripts/free_m.py; J=$DATA_DIR/runs/defect_fix/jt; AH=$DATA_DIR/runs/alpasim_proxy/arena/hold
B2=STANDFMXH-L18-s1; NTV0=NTV0-L18-s0; P1TAGS="R8K-L18-s2 R6DAG-L18-s1"; PRIO=${PRIO:-22}
RD=(--vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4)          # nuPlan reads: 4 a card (alpasim-read4)
mkdir -p $O/pool $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner nuplan-dag --name "$name" --tries ${TRIES:-1} --priority $PRIO --no-check \
     --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
case ${1:-} in
p1)
  T=(${2:-$P1TAGS}); ids=(); nch=$(python3 -c "import json; print(len(json.load(open('$AH/meta.json'))['chunks']))")
  for t in ${T[@]}; do [[ -f $DATA_DIR/runs/op_parity/runs/$t/ckpt-final.pt ]] || die "no $t"
    for c in $(seq 0 $((nch - 1))); do ids+=($(TRIES=2 sub nd-ar-$t-c$c "${RD[@]}" -- bash $JT aread $t off $c)) || exit 1; done
    for i in 0 1 2 3 4 5; do ids+=($(TRIES=2 sub nd-held-$t-$i "${RD[@]}" -- bash experiments/route_mem/scripts/rm_chain.sh read $t $i 6)) || exit 1; done; done
  s=$(sub nd-p1sum --after $(csv ${ids[@]}) --vram 0.1 --cpu 32 --ram 96 --timeout-h 4 -- bash $ME p1sum "${T[*]}") || exit 1
  echo "$(date '+%F %T') p1 (code $(git rev-parse --short HEAD)): ${T[*]}: reads ${ids[0]} .. ${ids[-1]} (${#ids[@]}) p1sum $s" | tee -a $O/stages.txt ;;
p1sum)  # the arena extract (native_gap rule) of the PAI tags + B2 + NTV0 -> paired arena / held 460 tables, zero classes, lateral-zero split
  T=(${2:-$P1TAGS}); X=$O/p1/ext; rd=(); for t in ${T[@]} $B2 $NTV0; do rd+=($t=$J/arena/$t@off); done
  $AP $FM ext --read ${rd[@]} --out $X --workers 32 > $O/p1/ext.log 2>&1 || die "extract (log $O/p1/ext.log)"
  for t in ${T[@]} $NTV0; do $V $FM arena --src $X --a $t --b $B2 --ref $NTV0 --out $O/p1/arena_$t.json > /dev/null || die "arena $t"; done
  $V -c "import pandas as pd; w = pd.read_parquet('$X/win.parquet'); w[['tag', 'id', 'score', 'type']].to_csv('$RES/p1_arena_windows.csv.gz', index=False)" || die windows
  $TR $PY held --tags ${T[@]} $NTV0 --refs $B2 $NTV0 --out $O/p1/held.json || die "held"
  $AP $PY latsplit --ext $X --tags ${T[@]} $B2 $NTV0 --out $O/p1/latsplit.csv || die "latsplit"
  $V $PY p1md --root $O/p1 --tags ${T[@]} --b2 $B2 --ntv0 $NTV0 --out $RES/p1.md || die "p1md"
  date '+%F %T' > $O/P1DONE ;;
wait)  # returns when $O/<marker> or VERDICT exists or one of this lane's jobs failed / was cancelled
  M=${2:?marker}
  until [[ -f $O/VERDICT || -f $O/$M ]] || ! $V - $O/jobs.txt <<'PY'
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
  cat $O/VERDICT 2>/dev/null; [[ -f $O/$M ]] && echo "$M $(cat $O/$M)"; tail -3 $O/stages.txt; true ;;
*) sed -n 2,8p "$0"; exit 2 ;;
esac
