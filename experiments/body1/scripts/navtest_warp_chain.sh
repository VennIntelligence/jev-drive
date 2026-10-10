#!/usr/bin/env bash
# Re-read of the navtest own-plan numbers of decisions 234 / 244 on the checkpoints' own front protocol (warp): until 2026-10-10 bd4_g3.plans
# read lb_navtest on GIMM frames for warp-trained checkpoints (results/navtest_warp.md). CPU only, no training: plans come from jevdrive.bench's
# archived plan files where they exist (--bench-plans), a CPU forward on the warp cache otherwise. Nothing is written into the checkout:
# tables under $DATA_DIR/runs/body1/navtest_warp/{route,sdrop,check}, dumps as runs/body1/prog/ol_<name>_w.*. One chain of pool jobs
# (declared 0.5 GB, CUDA hidden), at most 28 cores at a time.
#   scripts/tmux_run.sh nvwarp bash experiments/body1/scripts/navtest_warp_chain.sh
# State: $DATA_DIR/runs/body1/navtest_warp/chain/{STATUS, DONE | ERROR, log.txt}. Rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/body1/navtest_warp
D=$O/chain; L=$D/pool; mkdir -p "$L" "$O/route" "$O/sdrop" "$O/check"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
CL="$DATA_DIR/envs/jevdrive/bin/python -m jevdrive.cl"
S=experiments/body1/scripts
SRC=experiments/body1/results/sdrop
status() { echo "$(date '+%F %T') body1 navtest-warp: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
# sub <name> <cores> <ram GB> cmd...: a CPU job of the pool
sub() { local n=$1 c=$2 r=$3 ld=$L/$1; shift 3; [[ -f $ld/DONE ]] && return
        local live; live=$($CL queue 2>/dev/null | awk -v n="nvw-$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && return; rm -f "$ld/ERROR"
        $CL submit --owner body1-navtest-warp --name "nvw-$n" --log-dir "$ld" --vram 0.5 --cpu "$c" --ram "$r" -- \
          env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS="$c" MKL_NUM_THREADS="$c" "$@" > /dev/null || die "submit $n"; }
waitjobs() { for n in "$@"; do until [[ -f $L/$n/DONE || -f $L/$n/ERROR ]]; do sleep 10; done; [[ -f $L/$n/ERROR ]] && die "job failed: $L/$n"; done; return 0; }

ARMS=(noA noB noC)
NEW8="$(for a in "${ARMS[@]}"; do echo -n "P2H10S-$a-F-s0 P2H10S-$a-F-s1 "; done)P2H10S-F-s0 P2H10S-F-s1"
REF8="P2H10-F-s0 P2H10-F-s1 P2H10-F-s0 P2H10-F-s1 P2H10-F-s0 P2H10-F-s1 P2H10-F-s0 P2H10-F-s1"
declare -A NEW=([a6_full]="P2H10S-F-s0 P2H10S-F-s1" [a6_s23]="P2H10S-F-s2 P2H10S-F-s3 P2H10S-F-s0"
                [a6_pilot]="P2H10S-P-s0 P2H10B-Pw3-s0 P2H10B-Pw3-noA-s0 P2H10B-P-Aon-s0" [a7_pilot]="P2H10R-Pb25-s0" [sdrop]="$NEW8" [chk]="P2H10S-P-s0 P2H10S-noC-F-s0")
declare -A REF=([a6_full]="P2H10-F-s0 P2H10-F-s1" [a6_s23]="P2H10-F-s2 P2H10-F-s3 P2H10-F-s0" [a6_pilot]="P2H10-P-s0" [a7_pilot]="P2H10-P-s0" [sdrop]="$REF8" [chk]="P2H10-P-s0 P2H10-F-s0")
declare -A CORES=([a6_full]=2 [a6_s23]=8 [a6_pilot]=6 [a7_pilot]=4 [sdrop]=2 [chk]=6)
out() { case $1 in sdrop) echo "$O/sdrop";; chk) echo "$O/check";; *) echo "$O/route";; esac; }

status "stage 1: plan dumps on warp frames (chk: a forward of four archived checkpoints, the identity check of the fixed reader)"
for k in "${!NEW[@]}"; do
  bp=--bench-plans; [[ $k == chk ]] && bp=
  sub dump-$k "${CORES[$k]}" 24 $PY $S/prog_ol.py states --set navtest --new ${NEW[$k]} --ref ${REF[$k]} --out "$(out $k)" --name ${k}_navtest_w $bp
done
waitjobs $(for k in "${!NEW[@]}"; do echo dump-$k; done)

status "stage 2: route tables, own-plan contact rates of the S-DROP arms"
for k in "${!NEW[@]}"; do
  sub route-$k 2 12 $PY $S/route_ol.py --name ${k}_navtest_w --set navtest --new ${NEW[$k]} --ref ${REF[$k]} --out "$(out $k)"
done
for a in "${ARMS[@]}"; do for s in 0 1; do
  sub g3-$a-s$s 2 12 $PY $S/bd4_g3.py --name sdrop_navtest_${a}_s$s --set navtest --new P2H10S-$a-F-s$s --ref P2H10-F-s$s P2H10S-F-s$s --out "$O/sdrop" --bench-plans
done; done
waitjobs $(for k in "${!NEW[@]}"; do echo route-$k; done) $(for a in "${ARMS[@]}"; do echo g3-$a-s0 g3-$a-s1; done)

status "stage 3: S-DROP report on the warp dump (hold, bench and replay inputs unchanged, copied from $SRC)"
cp $SRC/d_*_vs_*.csv $SRC/g3_sdrop_hold_*.json "$O/sdrop/" || die "copy inputs"
sub report 4 24 $PY $S/sdrop_report.py report --navtest-dump sdrop_navtest_w --out "$O/sdrop"
waitjobs report
status "done"; date '+%F %T' > "$D/DONE"
