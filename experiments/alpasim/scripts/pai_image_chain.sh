#!/usr/bin/env bash
# Verification chain of the PAI submission image on the Tokyo box (results/pai_image.md), one card (CARD, default 1), two phases:
#   build   the four candidate images (docker/build.sh TRACK=pai: RSG-L18-s0 default serve, + JEV_VHIST=holdlead, + JEV_ACC=1, RBSG2-L18-s0),
#           test.sh on each (organisers' limits, PAI probe), replicas.sh on the default (4 replicas on the card, as ec2 places them)
#   smoke   pai_smoke.sh, steering weight 10, each image as built (no driver env): the default and ACC images on the 6-scene list of the
#           ACC prereg's Tokyo smoke (same host: the old image + mounted HEAD files, pai_run.sh), then all four on chunk c007 of the
#           box gains-10 reads (29 scenes; native references = the box's pai_g10/<label>/runs/rsg-l18-s0_c007, compared on the Mac)
#   experiments/alpasim/scripts/pai_image_chain.sh build|smoke <out root>
# Lists in <out root>/lists/{six,c007}.txt. State: <out root>/<phase>.{STATUS,DONE,ERROR}, images in <out root>/images.txt.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); D=$here/../docker
phase=$1 O=$2; CARD=${CARD:-1}; export DATA_DIR=${DATA_DIR:-/data}
mkdir -p "$O"; rm -f "$O/$phase".{DONE,ERROR}
exec > >(tee -a "$O/$phase.log") 2>&1
st() { echo "$(date '+%F %T') $*" | tee "$O/$phase.STATUS"; }
die() { st "ERROR $*"; echo "$*" > "$O/$phase.ERROR"; exit 1; }
# label | build env
V=("rsg|TAG=RSG-L18-s0" "rsghl|TAG=RSG-L18-s0 VHIST=holdlead" "rsgacc|TAG=RSG-L18-s0 ACC=1" "rbsg2|TAG=RBSG2-L18-s0")
img() { awk -v l="$1" '$1 == l {print $2}' "$O/images.txt"; }
case $phase in
build)
  : > "$O/images.txt"
  for v in "${V[@]}"; do
    l=${v%%|*}; st "build $l (${v#*|})"
    env TRACK=pai ${v#*|} OUT="$O/build/$l" bash "$D/build.sh" || die "build $l"
    echo "$l $(cat "$O/build/$l/IMAGE") $(cut -d' ' -f1 "$O/build/$l/image.txt")" >> "$O/images.txt"
  done
  for v in "${V[@]}"; do
    l=${v%%|*}; st "test.sh $l"
    bash "$D/test.sh" "$(img "$l")" "$O/test/$l" "" "$CARD" || die "test $l"
  done
  st "replicas.sh rsg: 4 replicas on card $CARD"
  bash "$D/replicas.sh" "$(img rsg)" "$O/replicas/rsg" 4 "$CARD" || die "replicas" ;;
smoke)
  W=controller.gains.rel_front_steering_angle_weight=10
  S6=/data/runs/route_follow/pai_acc_smoke
  run() {  # label image list [ref]
    [[ -f $O/smoke/$1/DONE ]] && return 0
    st "smoke $1: $(grep -c . "$3") scenes"
    GPU=$CARD WIZ=$W bash "$D/pai_smoke.sh" "$2" "$O/smoke/$1" "$3" "${4:-}" || echo "smoke $1 failed" >> "$O/smoke.ERROR"; }
  run six_rsg "$(img rsg)" "$O/lists/six.txt" "$S6/off/runs/off-rsg-l18-s0_c000/sim/aggregate/results-summary.json"
  run six_acc "$(img rsgacc)" "$O/lists/six.txt" "$S6/acc/runs/acc-rsg-l18-s0_c000/sim/aggregate/results-summary.json"
  for l in rsg rsgacc rsghl rbsg2; do run "c007_$l" "$(img "$l")" "$O/lists/c007.txt"; done
  [[ -f $O/smoke.ERROR ]] && die "$(cat "$O/smoke.ERROR")" ;;
*) sed -n 2,10p "$0"; exit 2 ;;
esac
date > "$O/$phase.DONE"; st "$phase done"
