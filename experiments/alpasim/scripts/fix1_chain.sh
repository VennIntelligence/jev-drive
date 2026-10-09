#!/usr/bin/env bash
# Lane FIX1 (plans/2026-10-09-fix1-prereg.md): the serving switches of lib/serve_fix.py in the closed loop on the public nuPlan scenes, as
# GPU-pool jobs through ot2_loop.py (OT_LANE=fix1). One self-advancing chain:
#   scripts/tmux_run.sh fix1 bash experiments/alpasim/scripts/fix1_chain.sh        (box; from the repo, or from an exported tree with a COMMIT file)
#   1  P2H10-F-s0 on the 700 scenes with OT3's chunk lists: base, +a, +b, +a+b at this checkout -> report, registered lines, WINNER
#   2  (only when an arm is kept) base and the winning configuration on the other 791 scenes (ot3new, cf1fresh), and APY10m10-AB-s0
#      base / winning configuration on all 1491
# State: $DATA_DIR/runs/alpasim/fix1/{STATUS, DONE, ERROR, log.txt, results/}; the loops' own state under fix1/s1, fix1/s2.
set -uo pipefail
cd "$(dirname "$0")/../../.."
RA=$DATA_DIR/runs/alpasim; O=$RA/fix1; S=experiments/alpasim/scripts; mkdir -p "$O/results"; rm -f "$O/DONE" "$O/ERROR"
exec > >(tee -a "$O/log.txt") 2>&1
VPY=$PWD/.venv/bin/python
status() { echo "$(date '+%F %T') fix1: $*" | tee "$O/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$O/ERROR"; exit 1; }
flags() { case $1 in *-ab) echo "JEV_VCONT=1.0,JEV_LEAD=1" ;; *-a) echo "JEV_VCONT=1.0" ;; *-b) echo "JEV_LEAD=1" ;; *-c1) echo "JEV_BASE=1" ;; *-c2) echo "JEV_BASE=2" ;;
                       *-c1a) echo "JEV_BASE=1,JEV_VCONT=1.0" ;; *-c2a) echo "JEV_BASE=2,JEV_VCONT=1.0" ;; *-c1b) echo "JEV_BASE=1,JEV_LEAD=1" ;; *-c2b) echo "JEV_BASE=2,JEV_LEAD=1" ;; *) echo "" ;; esac; }
# Amendment 1: FIX1_ARMS="c1 c2 c1b c2b" FIX1_S1=s1c adds arms on the 700 scenes against the base of the first start (s1) and reads all arms.
ARMS=${FIX1_ARMS:-base a b ab}; S1=${FIX1_S1:-s1}; N7=fix1_700${FIX1_S1:+_$FIX1_S1}
export OT_LANE=fix1 OT_PRIO=13 OT2_MAX_ACTIVE=${FIX1_STACKS:-6}
T=P2H10-F-s0; A=APY10m10-AB-s0
status "stage 1 ($S1), code $(cat COMMIT 2>/dev/null || git rev-parse --short HEAD): $T $ARMS on the 700 scenes"
J=()
for arm in $ARMS; do f=$(flags "x-$arm"); J+=("P2H10-$arm:sh30:SH30_TAG=$T${f:+,$f}:$T"); done
python3 $S/ot2_loop.py $S1 "${J[@]}" || die "ot2_loop $S1 rc $? (see $O/$S1/ERROR)"
M1=("$O/s1/manifest.json"); for x in s1c $S1; do [[ $x != s1 && -f $O/$x/manifest.json && " ${M1[*]} " != *" $O/$x/manifest.json "* ]] && M1+=("$O/$x/manifest.json"); done
$VPY $S/fix1_report.py --manifest "${M1[@]}" --base P2H10-base --arms P2H10-a P2H10-b P2H10-ab P2H10-c1 P2H10-c2 P2H10-c1b P2H10-c2b P2H10-c1a P2H10-c2a --pair P2H10-ab:P2H10-a --pair P2H10-ab:P2H10-b \
    --pair P2H10-c1b:P2H10-c1 --pair P2H10-c2b:P2H10-c2 --pair P2H10-c2:P2H10-c1 \
    --ref "OT3 run=$RA/ot3/a/manifest.json:$T" --lists all --out "$O/results" --name $N7 --winner > "$O/report_700.log" 2>&1 || die "report 700 failed (see $O/report_700.log)"
W=$(cat "$O/results/WINNER"); status "stage 1 done, winning configuration: $W"
date '+%F %T' > "$O/STAGE1_DONE"
if [[ $W == none ]]; then
  status "no arm kept on the 700 scenes: stage 2 is not run (registered stop rule)"; date '+%F %T' > "$O/DONE"; exit 0
fi
arm=${W#P2H10-}; f=$(flags "x-$arm")
status "stage 2: base and $arm on the other 791 scenes; $A base / $arm on 1491"
J=()
for l in ot3new cf1fresh; do J+=("P2H10-base@$l:sh30:SH30_TAG=$T:$T:$l" "P2H10-$arm@$l:sh30:SH30_TAG=$T,$f:$T:$l"); done
J+=("APY-base:ap2:AP2_TAG=$A:$A" "APY-$arm:ap2:AP2_TAG=$A,$f:$A")
for l in ot3new cf1fresh; do J+=("APY-base@$l:ap2:AP2_TAG=$A:$A:$l" "APY-$arm@$l:ap2:AP2_TAG=$A,$f:$A:$l"); done
python3 $S/ot2_loop.py s2 "${J[@]}" || die "ot2_loop s2 rc $? (see $O/s2/ERROR)"
M=("${M1[@]}" "$O/s2/manifest.json")
$VPY $S/fix1_report.py --manifest "${M[@]}" --base P2H10-base --arms "P2H10-$arm" --lists ot3new cf1fresh --out "$O/results" --name fix1_791 > "$O/report_791.log" 2>&1 || die "report 791"
$VPY $S/fix1_report.py --manifest "${M[@]}" --base P2H10-base --arms "P2H10-$arm" --lists all ot3new cf1fresh --out "$O/results" --name fix1_1491 > "$O/report_1491.log" 2>&1 || die "report 1491"
$VPY $S/fix1_report.py --manifest "${M[@]}" --base APY-base --arms "APY-$arm" --lists all --out "$O/results" --name fix1_apy_700 > "$O/report_apy_700.log" 2>&1 || die "report apy 700"
$VPY $S/fix1_report.py --manifest "${M[@]}" --base APY-base --arms "APY-$arm" --lists ot3new cf1fresh --out "$O/results" --name fix1_apy_791 > "$O/report_apy_791.log" 2>&1 || die "report apy 791"
$VPY $S/fix1_report.py --manifest "${M[@]}" --base APY-base --arms "APY-$arm" --lists all ot3new cf1fresh --out "$O/results" --name fix1_apy_1491 > "$O/report_apy_1491.log" 2>&1 || die "report apy 1491"
date '+%F %T' > "$O/DONE"; status "all done"
