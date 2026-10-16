#!/usr/bin/env bash
# ACC-GAP (plans/2026-10-16-acc-gap-prereg.md): ACC + the following-gap calibration (serve_fix JEV_GAP) on RSG-L18-s0, PAI, box pool,
# paired against the stored ACC read ($DATA_DIR/runs/route_follow/pai_g10/acc). Repo root on the box; ids in $O/jobs.txt.
#   rf_accgap.sh submit   stage 1: the 37-scene following subset (results/acc_gap/stage1.txt) with the arm (s1) and with plain ACC on the
#                         same chunk lists (s1acc, a same-list noise reference, descriptive) -> gate job (--after all four chunks)
#   rf_accgap.sh gate     (job body) stage-1 lines (rf_accgap.py lines --stage 1); only if they hold: stage 2 = all 281 on the ACC read's
#                         own chunk lists (s2) -> report job (--after its 8 chunks)
#   rf_accgap.sh report   (job body) stage-2 lines + rf_pai281 table (exam / control / stop17) against the ACC read
# Env: PRIO (14), GAP (the prereg's value).
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}
O=$DATA_DIR/runs/route_follow/acc_gap/read; A=$DATA_DIR/runs/route_follow/pai_g10/acc; S=experiments/route_follow/scripts; T=RSG-L18-s0
GAP=${GAP:-0.7,2.7}; WIZ=controller.gains.rel_front_steering_angle_weight=10; V=.venv/bin/python; PRIO=${PRIO:-14}
ARM="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=$GAP"; ACC="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1"
ids() { awk '{print $1}' "$1/jobs.txt" | paste -sd, -; }
case ${1:-} in
submit)
  [[ ! -e $O/s1 ]] || { echo "$O/s1 exists" >&2; exit 1; }
  mkdir -p $O/pool
  CHUNK=19 SERVE="$ARM" WIZ=$WIZ PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $O/s1 $T $S/../results/acc_gap/stage1.txt || exit 1
  CHUNK=19 SERVE="$ACC" WIZ=$WIZ PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $O/s1acc $T $S/../results/acc_gap/stage1.txt || exit 1
  g=$($V -m jevdrive.cl submit --owner accgap --name accgap-gate --tries 1 --priority $PRIO --no-check --vram 0.5 --cpu 2 --ram 8 --timeout-h 1 \
      --after "$(ids $O/s1),$(ids $O/s1acc)" --log-dir $O/pool/gate -- bash $S/rf_accgap.sh gate < /dev/null | awk '{print $NF}')
  [[ -n $g ]] || { echo "gate submit failed" >&2; exit 1; }; echo "$g accgap-gate" | tee -a $O/jobs.txt ;;
gate)
  $V $S/rf_accgap.py lines --stage 1 --arm $O/s1 --ref $A --list $S/../results/acc_gap/stage1.txt --extra accre=$O/s1acc --out $O/stage1
  rc=$?; touch $O/GATE_DONE; (( rc == 0 )) || { echo "stage 1 lines do not hold (rc $rc): no stage 2"; touch $O/STAGE2_DONE; exit 0; }
  CHUNKS_FROM=$A/chunks SERVE="$ARM" WIZ=$WIZ PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $O/s2 $T || exit 1
  r=$($V -m jevdrive.cl submit --owner accgap --name accgap-report --tries 1 --priority $PRIO --no-check --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 \
      --after "$(ids $O/s2)" --log-dir $O/pool/report -- bash $S/rf_accgap.sh report < /dev/null | awk '{print $NF}')
  [[ -n $r ]] || { echo "report submit failed" >&2; exit 1; }; echo "$r accgap-report" | tee -a $O/jobs.txt ;;
report)
  $V $S/rf_accgap.py lines --stage 2 --arm $O/s2 --ref $A --out $O/stage2; rc=$?
  $V $S/rf_pai281.py --rows acc=$A gap=$O/s2 --lists exam=experiments/alpasim/results/pai_exam/hard_exam.txt control=experiments/alpasim/results/pai_exam/control.txt \
     stop17=experiments/route_follow/results/lon_diag/route_end_scenes.txt --vs acc --ref acc --out $O/stage2_table.md
  echo "stage 2 lines rc $rc"; touch $O/STAGE2_DONE ;;
*) sed -n '2,10p' "$0"; exit 2 ;;
esac
