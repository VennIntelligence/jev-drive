#!/usr/bin/env bash
# Turning school P2, branch ceiling, on the Tokyo box (rf_branch.py, prereg experiments/route_follow/plans/2026-10-13-p2-branch-prereg.md):
# per handover arm one experiments/onpol_pai/scripts/onp_pass2.sh run (MODE=driver: onp_log_driver.py, the ego on the log up to the
# arm's decision k of each scene, the policy from k on via ONP_SWITCH), arms one after the other on the given cards.
#   rf_branch.sh <out root> <plan dir of rf_branch.py plan> [arm ...]          arms: d40 d20 d10 d0 (default all four)
# Env: CARDS ("1"; the worker waits until a card shows <= 1 GB), TAG (RSG-L18-s0), CONC (4), CHUNK (12 / number of cards),
#   SERVE ("-e JEV_VCONT=1.0 -e JEV_LEAD=1", the served configuration of the 281 reads), WIZ (gains 20), CODE (this lane's own clone,
#   /data/runs/route_night/p2/code: onp_pass2.sh copies the driver into it). Out: <out root>/<arm>/ as onp_pass2.sh; read with rf_branch.py read.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
O=$(mkdir -p "$1" && cd "$1" && pwd) PLAN=$(cd "$2" && pwd); shift 2
arms=${*:-d40 d20 d10 d0}; CARDS=${CARDS:-1}; nc=$(wc -w <<< "$CARDS")
export CARDS CONC=${CONC:-4} CHUNK=${CHUNK:-$(( (12 + nc - 1) / nc ))} WIZ=${WIZ-controller.gains.rel_front_steering_angle_weight=20}
export CODE=${CODE:-/data/runs/route_night/p2/code} MODE=driver
for a in $arms; do
  [[ -f $PLAN/sw_$a.txt ]] || { echo "no $PLAN/sw_$a.txt" >> "$O/ERROR"; continue; }
  echo "$(date '+%F %T') arm $a on cards $CARDS" | tee -a "$O/log.txt"
  SERVE="${SERVE--e JEV_VCONT=1.0 -e JEV_LEAD=1} -v $PLAN/sw_$a.txt:/switch.txt:ro -e ONP_SWITCH=/switch.txt" \
    bash "$here/../../onpol_pai/scripts/onp_pass2.sh" "$O/$a" "${TAG:-RSG-L18-s0}" "$PLAN/scenes.txt" > "$O/$a.out" 2>&1 \
    || echo "arm $a: $(tail -1 "$O/$a/ERROR" 2>/dev/null)" >> "$O/ERROR"
done
[[ -f $O/ERROR ]] && exit 1; date > "$O/DONE"
