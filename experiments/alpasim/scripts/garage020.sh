#!/usr/bin/env bash
# GARAGE020: the host's reference baseline py123d_garage v0.2.0 (ltf-base-joint64-68M posttrain/model_0.pth, conditional, their driver as
# shipped: 500 ms replanning, no post-processing, no route memory) on our PAI 281 exam, Harmonizer OFF, on NTV0-L18-s0's ACC + GAP chunk table
# (the simulator repeats a scene only within an identical list, so the scenes pair). Harness = our PAI box reads (pai_box.sh: AlpaSim 0bb4c4b,
# +e2e_challenge=dev, ALPASIM_FAST=1, CONC 4, steering weight 10) plus their driver config's skip_driver_during_force_gt=true (also the ec2
# leaderboard preset's value); their 2 Hz camera (camera_timing patch) is not applied: the driver gets the preset's 10 Hz frames and plans every
# 500 ms from 1.7 s. Every rollout.asl is kept (KEEP_ASL=1) for the fixed-corridor rescoring. Box, repo root; every job through the pool.
#   garage020.sh smoke            1-scene smoke (first scene of c000) -> $O/smoke
#   garage020.sh read [chunks]    chunks of the 281 read (default all eight c000..c007); a later call adds the rest (finished chunks are
#                                 skipped). Once all eight are submitted it chains: pai_fix tax -> facts, rescore old + new (CPU), report -> $O/DONE
#   garage020.sh wait <marker dir|all> one run_in_background waiter: DONE / ERROR of a pai_box run dir, or $O/DONE (all: $O/DONE and
#                                 $O/DONE_u0), or a failed / cancelled job in $O/jobs.txt (the newest id of each name)
#   GV=u0 garage020.sh smoke|read|wait: the unconditional variant (their midtrain stage, midtrain/model_0.pth, no target points): own dirs
#                                 (read/GARAGE020-u0, rescore_u0, results_u0, DONE_u0) and job names; GV=c0 (default) = conditional posttrain
#   job bodies: job-rescore <old|new> | job-report (GV from the job env)
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/alpasim/garage_v020; ME=experiments/alpasim/scripts/garage020.sh; PY=experiments/alpasim/scripts/garage020.py; PRIO=${PRIO:-14}
CH=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks; GV=${GV:-c0}; T=GARAGE020-$GV; R=$O/read/$T; F_=$DATA_DIR/runs/pai_fix/facts
X=$([[ $GV == c0 ]] || echo _$GV); N=$([[ $GV == c0 ]] && echo g020 || echo g020$GV)   # c0 keeps the first read's paths and job names
AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; NEW=$DATA_DIR/third_party/alpasim-e2e-1b024f0/src/eval/src
CK=$DATA_DIR/checkpoints/py123d_garage/v0.2.0/ltf-base-joint64-68M
CK=$DATA_DIR/checkpoints/py123d_garage/v0.2.0/ltf-base-joint64-68M; CKPT=$CK/$([[ $GV == u0 ]] && echo midtrain || echo posttrain)/model_0.pth
WIZ="controller.gains.rel_front_steering_angle_weight=10 runtime.simulation_config.skip_driver_during_force_gt=true"
mkdir -p $O/pool
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner garage020 --name "$name" --tries 1 --priority $PRIO \
     --no-check --log-dir $O/pool/$name --env GV=$GV "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
box() {  # box <out dir>: pai_box.sh submit with the garage driver, prints the new chunk job ids
  env DRV=garage020 SERVE="GARAGE_CKPT=$CKPT" WIZ="$WIZ" CONC=4 ALPASIM_FAST=1 KEEP_ASL=1 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $1 $T > $1/submit.txt \
    || die "pai_box submit $1"; awk '$2 ~ /^pai-/ {print $1, $2}' $1/submit.txt | tee -a $O/jobs.txt | awk '{print $1}'; }
case ${1:-} in
smoke)
  S=$O/smoke$X; [[ ! -e $S ]] || die "$S exists"; mkdir -p $S/chunks; head -1 $CH/c000.txt > $S/chunks/s000.txt
  [[ -f $CKPT && -f $DATA_DIR/runs/alpasim/garage_v020/setup/DONE ]] || die "setup_garage020.sh has not finished"
  ids=$(box $S); echo "$(date '+%F %T') smoke (code $(git rev-parse --short HEAD)): $ids" | tee -a $O/stages.txt ;;
read)
  mkdir -p $R/chunks; for c in ${2:-c000 c001 c002 c003 c004 c005 c006 c007}; do cp -n $CH/$c.txt $R/chunks/ || die "no chunk $c"; done
  ids=$(box $R); echo "$(date '+%F %T') read $(ls $R/chunks | tr '\n' ' ')(code $(git rev-parse --short HEAD)): $(echo $ids)" | tee -a $O/stages.txt
  if [[ $(ls $R/chunks/*.txt | wc -l) == 8 ]]; then
    A=$(echo $ids | tr ' ' ,)
    t=$(PRIO=$PRIO AFTER=$A bash experiments/pai_fix/scripts/pai_fix.sh tax $T-281off $R | tail -1) && [[ -n $t ]] || die "tax"; echo "$t pai_fix-tax$X" >> $O/jobs.txt
    ro=$(sub $N-rescore-old ${A:+--after $A} --vram 0.1 --cpu 8 --ram 40 --timeout-h 3 -- bash $ME job-rescore old) || exit 1
    rn=$(sub $N-rescore-new ${A:+--after $A} --vram 0.1 --cpu 8 --ram 40 --timeout-h 3 -- bash $ME job-rescore new) || exit 1
    rp=$(sub $N-report --after $t,$ro,$rn --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME job-report) || exit 1
    echo "$(date '+%F %T') chain: tax $t rescore $ro $rn report $rp" | tee -a $O/stages.txt
  fi ;;
job-rescore)
  [[ $2 == new ]] && export PYTHONPATH=$NEW
  $AP $PY rescore --variant $2 --read $R --root $O/rescore$X --workers 8 ;;
job-report)
  .venv/bin/python $PY report --facts $F_/$T-281off.csv --rescore $O/rescore$X --theirs $CK/results/nurec --out $O/results$X --label-title "$([[ $GV == u0 ]] && echo "unconditional, midtrain seed 0" || echo "conditional, posttrain seed 0")" > $O/results$X.log 2>&1 \
    || die "report (log $O/results$X.log)"
  date '+%F %T' > $O/DONE$X ;;
wait)
  d=${2:-$O}
  bad() { .venv/bin/python - $O/jobs.txt <<'PY'
import sys
from jevdrive.cl import pool as P
last = {}
for l in open(sys.argv[1]):
    if l.split():
        last[l.split()[1]] = l.split()[0]
jobs = P.snapshot()[0]["jobs"]
bad = [f"{n} {i} {jobs[i]['state']}: {jobs[i].get('why', '')}" for n, i in last.items() if jobs.get(i, {}).get("state") in ("failed", "cancelled")]
print("failed:", *bad, sep="\n  ", file=sys.stderr) if bad else None
sys.exit(0 if bad else 1)
PY
  }
  fin() { local n; [[ $d == all ]] && { [[ -f $O/DONE && -f $O/DONE_u0 ]] && return 0; bad; return; }
    [[ -f $d/DONE || -f $d/ERROR ]] && return 0; ls $d/runs/*/ERROR >/dev/null 2>&1 && return 0
    n=$(ls $d/chunks/*.txt 2>/dev/null | wc -l); (( n > 0 )) && [[ $(ls $d/runs/*/DONE 2>/dev/null | wc -l) == "$n" ]] && return 0; bad; }
  until fin; do sleep 300; done
  ls $d/runs/*/{DONE,ERROR} 2>/dev/null; ls $O/DONE* 2>/dev/null; tail -3 $O/stages.txt; true ;;
*) sed -n 2,15p $0; exit 2 ;;
esac
