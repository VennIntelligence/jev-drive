#!/usr/bin/env bash
# Harmonizer open-loop check (plans/2026-10-21-harmonizer-check.md): box, repo root; every GPU job through the pool at PRIO 16.
# The states of harm.py select are rendered three times along the human log (pai_rows' H path: onp_log_driver.py, A0 serve, NTV0-L18-s0
# in the loop) with identical chunk lists: OFFA, OFFB (re-render floor), ON (HARMONIZER=1, pai_native.sh); each -> onp_rows -> onp_cache build
# (the normal cache path, same encoder order) -> harm.py compare -> harm.py png.
#   harm.sh select              (job body) states.tsv + chunks/
#   harm.sh smoke               submit c000 (one scene) for the three variants
#   harm.sh smoketail           rows -> cache per variant on c000, then compare on its states (the whole tail once, before chain)
#   harm.sh chain               submit the other chunks, then rows -> clean, cache per variant, compare -> png (DONE)
#   harm.sh wait                blocks until $O/DONE or a failed job of this lane (one run_in_background waiter)
#   job bodies: rows <V> | clean <V> | cache <V> | compare | png
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round7/harm; ME=experiments/round7/scripts/harm.sh; PY=experiments/round7/scripts/harm.py
TR=$DATA_DIR/envs/op-train/bin/python; AS=$DATA_DIR/third_party/alpasim; AP=$AS/.venv/bin/python; V=.venv/bin/python; T=NTV0-L18-s0; PRIO=${PRIO:-16}
LOGSERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7 PAI_DUMP=100000 DRV_PY=../../onpol_pai/lib/onp_log_driver.py"
WIZ="controller.gains.rel_front_steering_angle_weight=10 runtime.simulation_config.send_recording_ground_truth=true"
VARS="OFFA OFFB ON"
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner round7-harm --name "$name" --tries 1 --priority $PRIO --no-check \
     --log-dir $O/pool/$name.$(date +%m%d%H%M%S) "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
roll() {  # roll <V> <chunk>...: copy the chunk lists into the variant's read dir, pai_box.sh submit (skips done / pending chunks) -> new job ids
  local v=$1 r=$O/roll/$1 f; shift; mkdir -p $r/chunks; for c in "$@"; do cp -n $O/chunks/$c.txt $r/chunks/; done
  local e=(SERVE="$LOGSERVE" WIZ="$WIZ" KEEP_ASL=1 CONC=4 PRIO=$PRIO); [[ $v == ON ]] && e=(SERVE="$LOGSERVE HARMONIZER=1" WIZ="$WIZ" KEEP_ASL=1 CONC=4 PRIO=$PRIO VRAM=30 RAM=40)
  f=$r/submit.$(date +%m%d%H%M%S).txt; env "${e[@]}" bash experiments/alpasim/scripts/pai_box.sh submit $r $T > $f || die "roll submit $v"
  awk '$2 ~ /^pai-/ {print $1}' $f | paste -sd, -; }
case ${1:-} in
select) mkdir -p $O; $TR $PY select --out $O ;;
smoke)
  mkdir -p $O/pool; [[ -f $O/chunks/c000.txt ]] || die "no chunks: run select first"
  for v in $VARS; do echo "$v $(roll $v c000)"; done | tee -a $O/stages.txt ;;
smoketail)
  cc=(); for v in $VARS; do w=$(sub harm-rows0-$v --vram 0.5 --cpu 8 --ram 40 --timeout-h 1 -- bash $ME rows $v) || exit 1
    k=$(sub harm-cache0-$v --after $w --vram 20 --cpu 8 --ram 64 --timeout-h 1 -- bash $ME cache $v) || exit 1; cc+=($k); done
  m=$(sub harm-compare0 --after $(IFS=,; echo "${cc[*]}") --vram 12 --cpu 8 --ram 64 --timeout-h 1 -- bash $ME compare) || exit 1
  echo "$(date '+%F %T') smoketail: caches ${cc[*]} compare $m" | tee -a $O/stages.txt ;;
chain)
  ch=$(cd $O/chunks && ls c*.txt | sed 's/.txt$//'); git rev-parse --short HEAD >> $O/code.txt; cc=()
  for v in $VARS; do
    r=$(roll $v $ch) || exit 1; [[ -n $r ]] || die "roll $v: no new job"
    w=$(sub harm-rows-$v --after $r --vram 0.5 --cpu 8 --ram 40 --timeout-h 3 -- bash $ME rows $v) || exit 1
    sub harm-clean-$v --after $w --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $ME clean $v > /dev/null || exit 1
    k=$(sub harm-cache-$v --after $w --vram 20 --cpu 8 --ram 64 --timeout-h 2 -- bash $ME cache $v) || exit 1; cc+=($k)
  done
  m=$(sub harm-compare --after $(IFS=,; echo "${cc[*]}") --vram 12 --cpu 8 --ram 64 --timeout-h 2 -- bash $ME compare) || exit 1
  p=$(sub harm-png --after $m --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- bash $ME png) || exit 1
  echo "$(date '+%F %T') chain (code $(git rev-parse --short HEAD)): caches ${cc[*]} compare $m png $p" | tee -a $O/stages.txt ;;
rows)
  v=$2; G=$DATA_DIR/tmp/harm_stage/$v; mkdir -p $G $O/rows
  env PAI_DUMP=100000 JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD $AP experiments/onpol_pai/scripts/onp_rows.py rows --pass $v \
      --runs "$O/roll/$v/runs/*" --out $O/rows --stage $G --tag $T --jobs 8 || exit 1
  env PAI_DUMP=100000 JEV_REPO=$PWD PYTHONPATH=$PWD/experiments/alpasim/lib:$PWD $TR experiments/onpol_pai/scripts/onp_rows.py cold --pass $v --out $O/rows --stage $G --jobs 8 || exit 1
  rm -rf $G; n=$(ls $O/rows/$v/*.npz 2>/dev/null | wc -l); m=$(cat $O/roll/$v/chunks/*.txt | grep -c .); echo "$(date '+%F %T') rows $v: $n / $m scenes" | tee -a $O/stages.txt
  (( n == m )) || die "rows $v: $n of $m scenes" ;;
clean) r=$O/roll/$2; find $r/runs -name rollout.asl -delete; rm -rf $r/runs/*/sim/driver-logs/dump; echo "$(date '+%F %T') clean $2: $(du -sh $r | cut -f1) left" >> $O/stages.txt ;;
cache) $TR experiments/onpol_pai/scripts/onp_cache.py build --pass $2 --rows $O/rows/$2 --out $O/cache ;;
compare) $TR $PY compare --out $O --cache $O/cache --rows $O/rows ;;
png) $TR $PY png --out $O --rows $O/rows && date '+%F %T' > $O/DONE ;;
wait)
  until [[ -f $O/DONE ]] || ! $V - $O/jobs.txt $O/roll <<'PY'
import glob, sys
from jevdrive.cl import pool as P
ids = [l.split()[0] for l in open(sys.argv[1]) if l.split()] + [l.split()[0] for f in glob.glob(f"{sys.argv[2]}/*/jobs.txt") for l in open(f) if l.split()]
jobs = P.snapshot()[0]["jobs"]
bad = [f"{i} {jobs[i]['state']}" for i in ids if jobs.get(i, {}).get("state") in ("failed", "cancelled")]
print("failed:", *bad, sep="\n  ", file=sys.stderr) if bad else None
sys.exit(1 if bad else 0)
PY
  do sleep 600; done
  [[ -f $O/DONE ]] && echo "DONE $(cat $O/DONE)"; true ;;
*) sed -n 2,13p "$0"; exit 2 ;;
esac
