#!/usr/bin/env bash
# REAL-GAP pool jobs (pav_real.py). Box, repo root; GPU work only through the pool.
#   pav_real.sh render    the 50 exam scenes of results/real_sample.csv along the human log (onp_log_driver.py), NTV0-L18-s0 fed, every
#                         session dumped once per second; pai_box.sh, 2 chunks of 25 -> $RUN/render
#   pav_real.sh replay    one GPU pool job after the render chunks: pav_real.py replay -> $RUN/replay
#   pav_real.sh report    one CPU pool job after replay: pav_real.py report + fig
set -uo pipefail; cd "$(dirname "$0")/../../.."
RUN=$DATA_DIR/runs/pai_av/realgap; T=NTV0-L18-s0; PRIO=${PRIO:-13}; TR=$DATA_DIR/envs/op-train/bin/python
ME=experiments/pai_av/scripts/pav_real.sh; PY=experiments/pai_av/scripts/pav_real.py
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7 PAI_DUMP=100000 DRV_PY=../../onpol_pai/lib/onp_log_driver.py"
WIZ="controller.gains.rel_front_steering_angle_weight=10 runtime.simulation_config.send_recording_ground_truth=true"
mkdir -p $RUN/pool
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner realgap --name "$name" --tries 1 --priority $PRIO \
     --no-check --log-dir $RUN/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }
     echo "$id $name" >> $RUN/jobs.txt; echo "$id"; }
case ${1:-} in
render)
  awk -F, 'NR > 1 && $2 == "test281" {print $3}' experiments/pai_av/results/real_sample.csv > $RUN/exam50.txt
  [[ $(wc -l < $RUN/exam50.txt) == 50 ]] || { echo "expected 50 exam scenes" >&2; exit 1; }
  CHUNK=25 SERVE="$SERVE" WIZ="$WIZ" CONC=4 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $RUN/render $T $RUN/exam50.txt ;;
replay)
  after=$(awk '{print $1}' $RUN/render/jobs.txt | paste -sd,)
  sub rg-replay ${after:+--after $after} --vram 10 --cpu 12 --ram 60 --timeout-h 3 -- $TR $PY replay ;;
report)
  after=$(grep ' rg-replay$' $RUN/jobs.txt | tail -1 | cut -d' ' -f1)
  sub rg-report ${after:+--after $after} --vram 0.1 --cpu 16 --ram 40 --timeout-h 1 -- bash -c "$TR $PY report && $TR $PY fig" ;;
*) sed -n '2,6p' "$0"; exit 2 ;;
esac
