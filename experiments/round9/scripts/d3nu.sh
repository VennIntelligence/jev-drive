#!/usr/bin/env bash
# ROUND9 D3-NUPLAN: D3 = NT4D3-F-s0 (decisions 473 / 475, the clean PAI-trained model) read on nuPlan with decision 444's readout (arena held
# split 18 chunks native = jt.sh aread <tag> off, held 460 @18 = rm_chain.sh read), paired against the stored B2 (STANDFMXH-L18-s1, decision 391)
# and NTV0-L18-s0 reads (R8K-L18-s2 / R6DAG-L18-s1 from decision 444 as context). Native serve, no stop3 (decision 371). Box, repo root;
# every job through the pool on cards 2-5 (cards 0-1 hold other lanes' trainings); ids in $O/jobs.txt, stages in $O/stages.txt.
#   d3nu.sh launch     24 read jobs + sum (one submit)
#   d3nu.sh wait       one run_in_background waiter: $O/DONE or a failed / cancelled job of this lane
# Part 2 (d3f.py; lane dir $F = runs/round9/d3_fail): failure statistics + one clip per failure type for research/d3-failures/
#   d3nu.sh fail <pai|nu>  (nu: after DONE) d3f.py select -> one CPU job rendering the stored PAI asl, the camera reruns as a chain of single GPU jobs
#                      (PRIO 4, cards 2-5, PAI CONC 2), d3f.py collect after all; lane dir $F/<phase>; d3nu.sh failwait <phase> = the waiter
#   job bodies: sum | vread <phase> <list>
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round9/d3_nuplan; ME=experiments/round9/scripts/d3nu.sh; V=.venv/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
JT=experiments/defect_fix/scripts/jt.sh; RM=experiments/route_mem/scripts/rm_chain.sh; RES=experiments/round9/results/d3_nuplan
T=NT4D3-F-s0; REFS="STANDFMXH-L18-s1 NTV0-L18-s0"; CTX="R8K-L18-s2 R6DAG-L18-s1"; PRIO=${PRIO:-21}
RD=(--vram 20 --cpu 6 --ram 30 --timeout-h 2 --gpus 2,3,4,5 --pin-strict --env CL_HIST_KEY=alpasim-read4)   # nuPlan reads: 4 a card
mkdir -p $O/pool $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id
  id=$($V -m jevdrive.cl submit --owner round9-d3nu --name "$name" --tries ${TRIES:-1} --priority $PRIO --no-check --log-dir $O/pool/$name "$@" < /dev/null \
       | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
F=$DATA_DIR/runs/round9/d3_fail; FP=experiments/round9/scripts/d3f.py
SERVE_PAI="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ_PAI=controller.gains.rel_front_steering_angle_weight=10
waitjobs() {  # waitjobs <done marker> <jobs file>
  until [[ -f $1 ]] || ! $V - $2 <<'PY'
import sys
from jevdrive.cl import pool as P
last = {}
for l in open(sys.argv[1]):
    if l.split():
        last[l.split()[-1]] = l.split()[0]
jobs = P.snapshot()[0]["jobs"]
bad = [f"{n} {i} {jobs[i]['state']}: {jobs[i].get('why', '')}" for n, i in last.items() if jobs.get(i, {}).get("state") in ("failed", "cancelled")]
if bad:
    print("failed:", *bad, sep="\n  ", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
  do sleep 300; done; [[ -f $1 ]] && echo "DONE $(cat $1)"; true; }
case ${1:-} in
fail)
  ph=$2; F=$F/$ph; TR=$([[ $ph == pai ]] && echo pai || echo arena,held); export D3F_O=$F
  [[ $ph == pai || -f $O/DONE ]] || die "Part 1 not done"; [[ ! -e $F/jobs.txt ]] || die "$F/jobs.txt exists"; mkdir -p $F/pool $F/rerun
  $V $FP select --tracks $TR > $F/select.log 2>&1 || die "select (log $F/select.log)"; tail -1 $F/select.log
  P2=(--owner round9-d3fail --tries 1 --priority 4 --no-check); prev=""; ids=()
  fsub() { local n=$1; shift; local id; id=$($V -m jevdrive.cl submit "${P2[@]}" --name $n --log-dir $F/pool/$n "$@" < /dev/null | awk '{print $NF}') \
             && [[ -n $id ]] || die "submit $n"; echo "$id $n" >> $F/jobs.txt; echo $id; }
  a=$(fsub d3f-asl --vram 0.1 --cpu 4 --ram 24 --timeout-h 2 -- env D3F_O=$F $AP $FP asl) || exit 1; ids+=($a)
  for f in $F/lists/*.txt; do l=$(basename $f .txt)
    if [[ $l == pai_* ]]; then t=${l#pai_}
      out=$(cd $PWD && SERVE="$SERVE_PAI" WIZ=$WIZ_PAI CONC=2 ALPASIM_FAST=1 KEEP_VIDEO=1 PRIO=4 GPUS=2,3,4,5 AFTER=$prev \
            bash experiments/alpasim/scripts/pai_box.sh submit $F/rerun/$l $t $f) || die "pai_box $l: $out"
      j=$(awk '$2 ~ /^pai-/ {print $1}' <<< "$out" | tail -1); [[ -n $j ]] || die "pai_box $l: $out"; echo "$j $l" >> $F/jobs.txt
    else j=$(fsub d3f-$l --vram 20 --cpu 6 --ram 30 --timeout-h 2 --gpus 2,3,4,5 --pin-strict --env CL_HIST_KEY=alpasim-read4 ${prev:+--after $prev} \
             -- bash $ME vread $ph $l) || exit 1; fi
    prev=$j; ids+=($j); done
  c=$(fsub d3f-collect --after $(csv ${ids[@]}) --vram 0.1 --cpu 4 --ram 24 --timeout-h 1 -- env D3F_O=$F $AP $FP collect) || exit 1
  echo "$(date '+%F %T') fail (code $(git rev-parse --short HEAD)): asl $a, reruns ${ids[*]:1}, collect $c" | tee -a $F/stages.txt ;;
vread)  # vread <list>: a camera rerun of the list's windows, = jt.sh aread (arena, sh30hold, no hold, the shard cache) or rm_chain.sh read
        # (held 460: sh30, after160 cache, 18 decisions), plus eval.video.render_video=true; no prune (= jtn_diag.sh vread)
  F=$F/$2; l=$3; t=${l##*_}; D=$F/rerun/$l; LS=$F/lists/$l.txt; KEY=runtime.scene_provider.trajdata
  source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
  rm -rf $D; mkdir -p $D; rc=0
  case $l in
    held_*) env -u JEV_HOLD SH30_TAG=$t SH30_DUMP=0 bash experiments/alpasim/scripts/run.sh $D sh30 --scene-list $LS "${BASE[@]}" \
              "$KEY.cache_location=$DATA_DIR/datasets/alpasim_nuplan_long/after160" "$KEY.dataset.extra_params.num_timesteps_after=160" eval.video.render_video=true || rc=$? ;;
    arena_s*) k=${l#arena_s}; k=${k%%_*}
      env -u JEV_HOLD SH30_TAG=$t bash experiments/alpasim/scripts/run.sh $D sh30hold --scene-list $LS "${BASE[@]}" \
        "$KEY.cache_location=$DATA_DIR/datasets/alpasim_nuplan_arena/v1/s$k" eval.video.render_video=true || rc=$? ;;
    *) die "unknown list $l" ;;
  esac
  if [[ $rc -ne 0 ]]; then
    grep -q "exit code -9" "$D/native_summary.json" 2>/dev/null && { echo "runtime worker killed (-9): exit 137 for the pool's kill retry" >&2; exit 137; }
    die "run.sh rc $rc ($D)"; fi
  echo "$(find $D/rollouts -name '*.mp4' | wc -l) mp4" ;;
failwait) F=$F/$2; waitjobs $F/DONE $F/jobs.txt; tail -2 $F/stages.txt ;;
launch)
  [[ ! -e $O/DONE ]] || die "already ran"; [[ -s $DATA_DIR/runs/op_parity/runs/$T/ckpt-final.pt ]] || die "no $T"
  nch=$(python3 -c "import json; print(len(json.load(open('$DATA_DIR/runs/alpasim_proxy/arena/hold/meta.json'))['chunks']))"); [[ $nch == 18 ]] || die "chunks $nch"
  echo "$(git rev-parse --short HEAD) launch" >> $O/code.txt; R=()
  for c in $(seq 0 $((nch - 1))); do R+=($(TRIES=2 sub d3nu-ar-c$c "${RD[@]}" -- bash $JT aread $T off $c)) || exit 1; done
  for i in 0 1 2 3 4 5; do R+=($(TRIES=2 sub d3nu-held-$i "${RD[@]}" -- bash $RM read $T $i 6)) || exit 1; done
  s=$(sub d3nu-sum --after $(csv ${R[@]}) --vram 0.1 --cpu 16 --ram 64 --timeout-h 2 -- bash $ME sum) || exit 1
  echo "$(date '+%F %T') launch (code $(git rev-parse --short HEAD)): reads ${R[0]} .. ${R[-1]} (${#R[@]}) sum $s" | tee -a $O/stages.txt ;;
sum)
  K=(); for t in $T $REFS $CTX; do K+=($t@off); done
  $AP -m jevdrive.facts fill --bench arena --keys ${K[@]} || die "fill arena"
  $AP -m jevdrive.facts fill --bench held460 --keys $T $REFS $CTX || die "fill held460"
  $V -m jevdrive.facts table --bench arena --tags $T $CTX --refs $REFS --out $O/arena.md --title "nuPlan arena held, native: D3 (NT4D3-F-s0) vs B2 / NTV0" || die "arena table"
  $V -m jevdrive.facts table --bench held460 --tags $T $CTX --refs $REFS --out $O/held460.md --title "held 460 @18: D3 (NT4D3-F-s0) vs B2 / NTV0" || die "held table"
  X=(); for t in $T $REFS $CTX; do X+=($t=$t@off); done
  $AP -m jevdrive.facts ext --read "${X[@]}" --out $O/ext > $O/ext.log 2>&1 || die "ext (log $O/ext.log)"
  $AP -m jevdrive.facts latsplit --read "${X[@]}" --out $O/latsplit.csv || die "latsplit"
  $V -c "import pandas as pd; w = pd.read_parquet('$O/ext/win.parquet'); w[['tag', 'id', 'score', 'type']].to_csv('$O/arena_windows.csv.gz', index=False)" \
    || die "windows"
  cp $O/arena.md $O/arena.json $O/held460.md $O/held460.json $O/latsplit.csv $O/arena_windows.csv.gz $RES/ || die copy
  date '+%F %T' > $O/DONE ;;
wait)
  until [[ -f $O/DONE ]] || ! $V - $O/jobs.txt <<'PY'
import sys
from jevdrive.cl import pool as P
last = {}
for l in open(sys.argv[1]):
    if l.split():
        last[l.split()[-1]] = l.split()[0]
jobs = P.snapshot()[0]["jobs"]
bad = [f"{n} {i} {jobs[i]['state']}: {jobs[i].get('why', '')}" for n, i in last.items() if jobs.get(i, {}).get("state") in ("failed", "cancelled")]
if bad:
    print("failed:", *bad, sep="\n  ", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
  do sleep 300; done
  [[ -f $O/DONE ]] && echo "DONE $(cat $O/DONE)"; tail -3 $O/stages.txt; true ;;
*) sed -n 2,8p "$0"; exit 2 ;;
esac
