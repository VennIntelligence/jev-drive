#!/usr/bin/env bash
# READ-SUMMARIES (results/read_summaries.md): what the summaries after closed-loop reads spend their time on, and the facts-cache path
# (jevdrive/facts.py) against the old one on nuplan_dag Part 1. Box, repo root; every job a CPU job through the pool (PRIO 23).
#   rs_bench.sh submit     old (nd.sh p1sum's steps verbatim into a scratch dir, 32 cores) | chunk (one arena + one held 460 chunk alone,
#                          6 cores = a read job's) | fill (backfill of the reference reads, one parallel job) -> new (the cached summary
#                          on 4 cores + identity against the lane's stored p1 outputs and the old run) ; $B/DONE at the end
#   rs_bench.sh wait       one run_in_background waiter: $B/DONE or a failed job
#   job bodies: old | chunk | fill | new
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
B=$DATA_DIR/runs/facts/_bench; ME=experiments/cl_infra/scripts/rs_bench.sh; V=.venv/bin/python
TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
PY=experiments/nuplan_dag/scripts/nd.py; FM=experiments/defect_fix/scripts/free_m.py; J=$DATA_DIR/runs/defect_fix/jt
ND=$DATA_DIR/runs/nuplan_dag; RES=experiments/nuplan_dag/results
B2=STANDFMXH-L18-s1; NTV0=NTV0-L18-s0; T=(R8K-L18-s2 R6DAG-L18-s1)
REFS="STANDFMXH-L18-s1 STANDFMXH-L18-s0 NTV0-L18-s0 RSG-L18-s0 R8K-L18-s2 R6DAG-L18-s1"
mkdir -p $B/pool
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $B/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner cl-infra --name "$name" --tries 1 --priority 23 --no-check \
     --log-dir $B/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $B/jobs.txt; echo "$id"; }
el() { awk -v a=$1 -v b=$(date +%s%N) 'BEGIN {printf "%.1f", (b - a) / 1e9}'; }   # seconds since $1 (ns)
ts() { local n=$1; shift; local t0; t0=$(date +%s%N); "$@" > $W/$n.log 2>&1; local rc=$?
  printf '%s\t%s\trc %s\n' "$n" "$(el $t0)" $rc | tee -a $W/times.tsv; [[ $rc == 0 ]] || die "$n rc $rc ($W/$n.log)"; }
rd() { for t in ${T[@]} $B2 $NTV0; do echo $t=$J/arena/$t@off; done; }
case ${1:-} in
submit)
  rm -f $B/DONE; o=$(sub rs-old --vram 0 --cpu 32 --ram 96 --timeout-h 2 -- bash $ME old) || exit 1
  c=$(sub rs-chunk --vram 0 --cpu 6 --ram 30 --timeout-h 1 -- bash $ME chunk) || exit 1
  f=$(sub rs-fill --after $c --vram 0 --cpu 32 --ram 96 --timeout-h 2 -- bash $ME fill) || exit 1
  n=$(sub rs-new --after $o,$f --vram 0 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME new) || exit 1
  echo "$(date '+%F %T') submit (code $(git rev-parse --short HEAD)): old $o chunk $c fill $f new $n" | tee -a $B/stages.txt ;;
old)    # nd.sh p1sum's commands (code dfdc59d2..2aef63fd), outputs into $B/old
  W=$B/old; rm -rf $W; mkdir -p $W
  ts quick $TR $PY quick --tags ${T[@]} --refs $B2 $NTV0 --out $W/p1_quick.md --title "Part 1: PAI checkpoints on nuPlan"
  ts ext $AP $FM ext --read $(rd) --out $W/ext --workers 32
  for t in ${T[@]} $NTV0; do ts arena_$t $V $FM arena --src $W/ext --a $t --b $B2 --ref $NTV0 --out $W/arena_$t.json; done
  ts windows $V -c "import pandas as pd; w = pd.read_parquet('$W/ext/win.parquet'); w[['tag', 'id', 'score', 'type']].to_csv('$W/p1_arena_windows.csv.gz', index=False)"
  ts held $TR $PY held --tags ${T[@]} $NTV0 --refs $B2 $NTV0 --out $W/held.json
  ts latsplit $AP $PY latsplit --ext $W/ext --tags ${T[@]} $B2 $NTV0 --out $W/latsplit.csv
  ts p1md $V $PY p1md --root $W --tags ${T[@]} --b2 $B2 --ntv0 $NTV0 --out $W/p1.md ;;
chunk)  # one read chunk extracted alone on a read job's 6 cores (arena B2 c0, held 460 B2 c0), forced
  W=$B/chunk; rm -rf $W; mkdir -p $W
  ts arena_c0 $AP -c "import sys; sys.argv = ['facts', 'chunk', '--bench', 'arena', '--run', '$J/arena/$B2@off/c0/run', '--force', '--workers', '6']; from jevdrive.facts import main; main()"
  ts held_c0 $AP -c "import sys; sys.argv = ['facts', 'chunk', '--bench', 'held460', '--run', '$DATA_DIR/runs/route_mem/read/$B2/c0/run', '--force', '--workers', '6']; from jevdrive.facts import main; main()"
  cat $DATA_DIR/runs/facts/arena/$B2@off/c0/DONE $DATA_DIR/runs/facts/held460/$B2/c0/DONE > $W/done.json ;;
fill)   # backfill: the reference reads of the live nuPlan lanes, arena + held 460, one batch each
  W=$B/fill; mkdir -p $W
  ts arena $AP -m jevdrive.facts fill --bench arena --keys $REFS --workers 32
  ts held460 $AP -m jevdrive.facts fill --bench held460 --keys $REFS --workers 32 ;;
new)    # the cached p1sum on 4 cores: score-only tables first, then the extract from the cache, arena pairs in parallel, held, latsplit, p1md
  W=$B/new; rm -rf $W; mkdir -p $W; t0=$(date +%s%N)
  ts table_arena $V -m jevdrive.facts table --bench arena --tags ${T[@]} --refs $B2 $NTV0 --out $W/table_arena.md --title "Part 1"
  ts table_held $V -m jevdrive.facts table --bench held460 --tags ${T[@]} --refs $B2 $NTV0 --out $W/table_held.md --title "Part 1"
  ts quick $TR $PY quick --tags ${T[@]} --refs $B2 $NTV0 --out $W/p1_quick.md --title "Part 1: PAI checkpoints on nuPlan"
  ts ext $AP -m jevdrive.facts ext --read $(rd) --out $W/ext
  t1=$(date +%s%N); pids=()
  for t in ${T[@]} $NTV0; do $V $FM arena --src $W/ext --a $t --b $B2 --ref $NTV0 --out $W/arena_$t.json > $W/arena_$t.log 2>&1 & pids+=($!); done
  ts held $TR $PY held --tags ${T[@]} $NTV0 --refs $B2 $NTV0 --out $W/held.json
  ts latsplit $AP -m jevdrive.facts latsplit --read $(rd) --out $W/latsplit.csv
  for p in ${pids[@]}; do wait $p || die "arena rc $?"; done; printf 'arena x3 (parallel)\t%s\n' "$(el $t1)" | tee -a $W/times.tsv
  ts windows $V -c "import pandas as pd; w = pd.read_parquet('$W/ext/win.parquet'); w[['tag', 'id', 'score', 'type']].to_csv('$W/p1_arena_windows.csv.gz', index=False)"
  ts p1md $V $PY p1md --root $W --tags ${T[@]} --b2 $B2 --ntv0 $NTV0 --out $W/p1.md
  printf 'total\t%s\n' "$(el $t0)" | tee -a $W/times.tsv
  $V experiments/cl_infra/scripts/rs_check.py --new $W --old $B/old --lane $ND/p1 --lane-md $RES/p1.md --out $B/identity.json || die "identity"
  date '+%F %T' > $B/DONE ;;
wait)
  until [[ -f $B/DONE ]] || ! $V - $B/jobs.txt <<'PY'
import sys
from jevdrive.cl import pool as P
last = {}
for l in open(sys.argv[1]):
    if l.split():
        last[l.split()[1]] = l.split()[0]
jobs = P.snapshot()[0]["jobs"]
bad = [f"{n} {i} {jobs[i]['state']}: {jobs[i].get('why', '')}" for n, i in last.items() if jobs.get(i, {}).get("state") in ("failed", "cancelled")]
print(*bad, sep="\n", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
  do sleep 120; done; tail -3 $B/stages.txt; cat $B/DONE 2>/dev/null; true ;;
*) sed -n 2,9p "$0"; exit 2 ;;
esac
