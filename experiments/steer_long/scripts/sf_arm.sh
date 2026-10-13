#!/usr/bin/env bash
# SPIN-FIX: one arm on one scene list on Tokyo card 0 (RSG-L18-s0 unless TAG), from this checkout; a 2-scene smoke first unless NOSMOKE=1.
#   sf_arm.sh <name> <scene list> [driver env, e.g. "-e JEV_TRACK=full -e JEV_TRACK_CFG=amin=-7"]
# Env: GAINS (rel_front_steering_angle_weight, default 20; 5 = no override), TAG, CARD (0). Out: /data/runs/steer_long/spinfix/reads/<name>{_smoke,}.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); repo=$(cd "$here/../../.." && pwd)
N=$1 L=$2 X=${3:-}; O=/data/runs/steer_long/spinfix/reads; TAG=${TAG:-RSG-L18-s0}; G=${GAINS:-20}
WZ=""; [[ $G != 5 ]] && WZ="controller.gains.rel_front_steering_angle_weight=$G"
run() {  # out list
  CARDS=${CARD:-0} CHUNK=60 SERVE="-e JEV_VCONT=1.0 -e JEV_LEAD=1 $X" WIZ="$WZ" bash "$repo/experiments/alpasim/scripts/pai_eval.sh" "$1" "$TAG" "$2"
}
if [[ ${NOSMOKE:-0} != 1 ]]; then
  mkdir -p "$O/${N}_smoke"; head -2 "$L" > "$O/${N}_smoke/list.txt"
  run "$O/${N}_smoke" "$O/${N}_smoke/list.txt" || { echo "SMOKE FAILED"; exit 1; }
  grep -h "serving:" "$O/${N}_smoke"/runs/*/driver.log | head -1
  [[ -z $X ]] || grep -q '"track"' "$O/${N}_smoke"/runs/*/driver/drive.jsonl || { echo "SMOKE: no track records"; exit 1; }
fi
run "$O/$N" "$L" && echo "FULL DONE $N"
