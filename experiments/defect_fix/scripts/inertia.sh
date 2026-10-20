#!/usr/bin/env bash
# INERTIA (plans/2026-10-20-inertia-prereg.md): pool jobs on the box, repo root. Ids in $O/jobs.txt.
#   inertia.sh recede        supplement 1 (CPU job): A0's stored 10 Hz plans on the overrun / control approaches
#   inertia.sh smoke         cf on 3 scenes (GPU job) -> $O/smoke
#   inertia.sh all           cf (GPU) -> report (CPU, --after)
#   stage 1 (supplement 3; NTV0-L18-s0 own states = A0's serve and FAST setting, human states = pai_gap.sh's H read; dumps kept):
#   inertia.sh s1select      the scene list -> $O/s1/scenes.txt, results/inertia/scenes_s1.csv (+ recede with the stop controls)
#   inertia.sh s1one <scene> the 1-scene re-roll -> parity against A0 (exit 2 = fail; nothing else is submitted)
#   inertia.sh s1all         own + H reads (CHUNK 14) -> rows -> clean (rollout.asl, dumps) -> cf -> report, per read
#   job bodies: rows <pass> <read dir> | clean <read dir> <pass>
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/defect_fix/inertia; PY=experiments/defect_fix/scripts/inertia.py; TR=$DATA_DIR/envs/op-train/bin/python; mkdir -p $O/pool
S1=$O/s1; AS=$DATA_DIR/third_party/alpasim; A0SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10
LOGDRV=DRV_PY=../../onpol_pai/lib/onp_log_driver.py; GT=runtime.simulation_config.send_recording_ground_truth=true
pai() {  # pai <out> <serve> <wiz> <scene list | -> [CHUNKS_FROM]: the read with dumps kept -> its chunk job ids
  CHUNK=14 CHUNKS_FROM=${4:-} SERVE="$2 PAI_DUMP=100000" WIZ="$3" CONC=4 KEEP_ASL=1 ALPASIM_FAST=1 PRIO=${PRIO:-10} \
    bash experiments/alpasim/scripts/pai_box.sh submit $1 NTV0-L18-s0 ${5:-} > $1.submit.txt || { echo "pai submit $1 failed" >&2; exit 1; }
  awk '$2 ~ /^pai-/' $1.submit.txt >> $O/jobs.txt; awk '$2 ~ /^pai-/ {print $1}' $1.submit.txt | paste -sd,; }
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner defect-fix --name "$name" --tries 1 --priority ${PRIO:-10} \
     --no-check --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
case ${1:-} in
recede) sub inertia-recede --vram 0.1 --cpu 4 --ram 48 --timeout-h 1 -- $TR $PY recede ;;
smoke)  sub inertia-smoke --vram 24 --cpu 8 --ram 48 --timeout-h 1 -- bash -c "O_SMOKE=1 $TR $PY cf --limit 3 --workers 3" ;;
all)    c=$(sub inertia-cf --vram 24 --cpu 16 --ram 96 --timeout-h 2 -- $TR $PY cf) || exit 1
        sub inertia-report --after $c --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- $TR $PY report ;;
s1select) $TR $PY select && sub inertia-recede-s1 --vram 0.1 --cpu 4 --ram 48 --timeout-h 1 -- $TR $PY recede --controls $S1/stop_ctrl.txt ;;
s1one)
  mkdir -p $S1; echo "$2" > $S1/one.txt
  q=$(pai $S1/one "$A0SERVE" "$WIZ" "" $S1/one.txt) && [[ -n $q ]] || exit 1
  sub inertia-s1-parity --after $q --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- bash -c "$TR $PY parity --read $S1/one > $S1/parity_one.json" ;;
s1all)
  [[ -f $S1/scenes.txt ]] || { echo "no scene list" >&2; exit 1; }
  qo=$(pai $S1/own "$A0SERVE" "$WIZ" "" $S1/scenes.txt) && [[ -n $qo ]] || exit 1
  qh=$(pai $S1/hum "$A0SERVE $LOGDRV" "$WIZ $GT" $S1/own/chunks) && [[ -n $qh ]] || exit 1
  po=$(sub inertia-s1-parity-own --after $qo --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- bash -c "$TR $PY parity --read $S1/own > $S1/parity_own.json; true") || exit 1
  for P in OWN HUM; do R=$S1/$([[ $P == OWN ]] && echo own || echo hum); q=$([[ $P == OWN ]] && echo $qo || echo $qh); n=${P,,}
    r=$(sub inertia-s1-rows-$n --after $q --vram 0.5 --cpu 24 --ram 96 --timeout-h 3 -- bash $0 rows $P $R) || exit 1
    sub inertia-s1-clean-$n --after $r --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $0 clean $R $P > /dev/null || exit 1
    c=$(sub inertia-s1-cf-$n --after $r --vram 24 --cpu 16 --ram 96 --timeout-h 2 -- $TR $PY cf --rows $S1/rows/$P --name s1_$n --gate-tag NTV0-L18-s0) || exit 1
    sub inertia-s1-report-$n --after $c --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- $TR $PY report --name s1_$n > /dev/null || exit 1; done
  echo "own $qo hum $qh parity $po" ;;
rows)   # rows <pass> <read dir>: onp_rows rows + cold (as stoppt.sh rows) -> $S1/rows/<pass>/<scene>.npz
  STG=$DATA_DIR/tmp/inertia_$2; mkdir -p $STG $S1/rows
  env PAI_DUMP=100000 JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD $AS/.venv/bin/python experiments/onpol_pai/scripts/onp_rows.py rows --pass $2 \
      --runs "$3/runs/*" --out $S1/rows --stage $STG --tag NTV0-L18-s0 --jobs 24 || exit 1
  env PAI_DUMP=100000 JEV_REPO=$PWD PYTHONPATH=$PWD/experiments/alpasim/lib:$PWD $TR experiments/onpol_pai/scripts/onp_rows.py cold --pass $2 --out $S1/rows --stage $STG --jobs 24 || exit 1
  env PYTHONPATH=$PWD/experiments/alpasim/lib:$PWD $TR experiments/onpol_pai/scripts/onp_rows.py report --pass $2 --out $S1/rows > $S1/rows/report_$2.log 2>&1; tail -3 $S1/rows/report_$2.log; rm -rf $STG ;;
clean)  # clean <read dir> <pass>: after the row files exist, the rollout.asl files and the frame dumps go
  n=$(ls $S1/rows/$3/*.npz 2>/dev/null | wc -l); [[ $n -gt 0 ]] || { echo "no row files for $3: nothing removed" >&2; exit 1; }
  m=$(find $2/runs -name rollout.asl | wc -l); find $2/runs -name rollout.asl -delete; rm -rf $2/runs/*/sim/driver-logs/dump
  echo "$3: $n row files; removed $m rollout.asl and the frame dumps; $(du -sh $2 | cut -f1) left" ;;
*) sed -n 2,12p $0; exit 2 ;;
esac
