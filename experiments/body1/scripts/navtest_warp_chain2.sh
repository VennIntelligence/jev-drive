#!/usr/bin/env bash
# Second part of the navtest re-read on warp frames (results/navtest_warp.md; first part: navtest_warp_chain.sh, which must be DONE): the reads
# of decisions 229 / 230 / 232 / 236 that the first chain does not cover. G3 (b) of Amendment 5 (P2H10B-F) and Amendment 6 (P2H10S-F, seeds 0-3),
# the diagnosis dump (P2H10B-F against P2H10-F), the SH30 look, then the gate tables (shape_gate.py, route_gate.py) and the closed-loop tables that
# group scenes by the base plan at the navtest token (prog_cl.py, shape_cl_report.py). CPU only, no training; plans from jevdrive.bench's archive
# where it exists (--bench-plans), a CPU forward on the warp cache otherwise. Nothing is written into the checkout: tables under
# $DATA_DIR/runs/body1/navtest_warp/{loss,prog,sh30s,shape,route,prog_cl,shape_cl}. At most 28 cores at a time.
#   scripts/tmux_run.sh nvwarp2 bash experiments/body1/scripts/navtest_warp_chain2.sh
# State: $DATA_DIR/runs/body1/navtest_warp/chain2/{STATUS, DONE | ERROR, log.txt}. Rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/body1/navtest_warp
D=$O/chain2; L=$D/pool; mkdir -p "$L" "$O"/{loss,prog,sh30s,shape,prog_cl,shape_cl}; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
CL="$DATA_DIR/envs/jevdrive/bin/python -m jevdrive.cl"
S=experiments/body1/scripts
status() { echo "$(date '+%F %T') body1 navtest-warp 2: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
# sub <name> <cores> <ram GB> cmd...: a CPU job of the pool
sub() { local n=$1 c=$2 r=$3 ld=$L/$1; shift 3; [[ -f $ld/DONE ]] && return
        local live; live=$($CL queue 2>/dev/null | awk -v n="nvw2-$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && return; rm -f "$ld/ERROR"
        $CL submit --owner body1-navtest-warp --name "nvw2-$n" --log-dir "$ld" --vram 0.5 --cpu "$c" --ram "$r" -- \
          env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS="$c" MKL_NUM_THREADS="$c" "$@" > /dev/null || die "submit $n"; }
waitjobs() { for n in "$@"; do until [[ -f $L/$n/DONE || -f $L/$n/ERROR ]]; do sleep 10; done; [[ -f $L/$n/ERROR ]] && die "job failed: $L/$n"; done; return 0; }

[[ -f $O/chain/DONE ]] || die "navtest_warp_chain.sh is not done"

status "stage 1: dumps and G3 (b) on warp frames"
B01="P2H10-F-s0 P2H10-F-s1"
sub dump-prog 2 24 $PY $S/prog_ol.py states --set navtest --out "$O/prog" --name navtest_w --bench-plans
sub dump-sh30s 2 24 $PY $S/prog_ol.py states --set navtest --new SH30-F-s0 SH30-F-s1 P2H10S-F-s0 P2H10S-F-s1 --ref $B01 $B01 --out "$O/sh30s" --name sh30s_look_navtest_w --bench-plans
for s in 0 1; do
  sub g3-a5-s$s 2 12 $PY $S/bd4_g3.py --name a5_full_navtest_s$s --set navtest --new P2H10B-F-s$s --ref P2H10-F-s$s --out "$O/loss" --bench-plans
done
for s in 0 1 2 3; do
  sub g3-a6-s$s $(( s < 2 ? 2 : 8 )) 12 $PY $S/bd4_g3.py --name a6_full_navtest_s$s --set navtest --new P2H10S-F-s$s --ref P2H10-F-s$s --out "$O/loss" --bench-plans
done
waitjobs dump-prog dump-sh30s g3-a5-s0 g3-a5-s1 g3-a6-s0 g3-a6-s1 g3-a6-s2 g3-a6-s3

status "stage 2: gate tables and the closed-loop tables grouped by the base plan (hold-log, trainer and loop inputs unchanged)"
C=$DATA_DIR/runs/body1/cl
sub gate-shape-pilot 2 8 $PY $S/shape_gate.py pilot --nav _w --out "$O/shape"
sub gate-g3e 2 8 $PY $S/shape_gate.py g3e --nav _w --out "$O/shape"
sub gate-g3e-s23 2 8 $PY $S/shape_gate.py g3e --seeds 2 3 --name a6_s23 --outname g3e_s23 --nav _w --out "$O/shape"
sub gate-route 2 8 $PY $S/route_gate.py pilot --arm P2H10R-Pb25-s0 --nav _w --out "$O/route"
sub cl-prog 4 16 $PY $S/prog_cl.py tables --ol ol_navtest_w --out "$O/prog_cl"
sub cl-shape 4 16 $PY $S/shape_cl_report.py --base-man "$DATA_DIR/runs/alpasim/tr1/a/manifest.json" "$C/base23/manifest.json" --arm-man "$C/shape-cl/manifest.json" \
  --loss-man "$C/loss-man.json" --out "$O/shape_cl" --ol ol_navtest_w
waitjobs gate-shape-pilot gate-g3e gate-g3e-s23 gate-route cl-prog cl-shape
status "done"; date '+%F %T' > "$D/DONE"
