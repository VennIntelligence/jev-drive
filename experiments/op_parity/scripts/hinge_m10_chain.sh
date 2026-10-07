#!/usr/bin/env bash
# op_parity hinge-m10 (plans/2026-10-08-hinge-m10-prereg.md): SH30 recipe with hinge lambda 30 / margin 1.0 at full scale, 2 seeds (arm SH30M10),
# scored on navtest / navhard (G) / HUGSIM 64 spec_plan_smooth, plus replay / raw-plan geometry, vs SH30 and P2H10.
# One self-advancing chain in tmux jev (scripts/tmux_run.sh hinge-m10 experiments/op_parity/scripts/hinge_m10_chain.sh). Every GPU / CPU job through the pool.
# State: $DATA_DIR/runs/op_parity/hinge_m10/chain/{STATUS, DONE, ERROR, log.txt, jobs.txt}; results under .../results/. Rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/op_parity/hinge_m10
D=$O/chain; mkdir -p "$D" "$O/results"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
NAV=$DATA_DIR/envs/navsim2/bin/python
VPY=$PWD/.venv/bin/python
CL="$DATA_DIR/envs/jevdrive/bin/python -m jevdrive.cl"
S=experiments/op_parity/scripts
B=("$VPY" -m jevdrive.bench)
L=$D/pool
R=$O/results
HF="--hinge-lam 30 --hinge-margin 1.0"
K=12 FULL=$(for i in $(seq 0 $((K - 1))); do echo -n "navtrain_full.s${i}of$K "; done)
FULL_SPLIT=navsim/op-parity-full
LIST=experiments/hugsim/scripts/derot_all64.txt
status() { echo "$(date '+%F %T') op_parity hinge-m10: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && return; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 20; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; return 0; }

status "train SH30M10-F-s0 / s1"
smoke_f="$PY $S/pp_train.py --arm P2 --frames warp --host --data navtrain_full.s0of12 --split $FULL_SPLIT --steps 3 --batch 16 --eval-every 3 $HF --tag smoke-m10"
for s in 0 1; do
  sub m10-t-s$s $L/t-s$s --train --vram 40 --cpu 6 --ram 40 --preflight "$smoke_f" -- $PY $S/pp_train.py --arm P2 --seed $s --frames warp --host \
      --data $FULL --split $FULL_SPLIT --steps 10000 --batch 128 --warmup 300 --eval-every 1000 $HF --tag SH30M10-F-s$s
done
waitdirs $L/t-s0 $L/t-s1
for s in 0 1; do $PY $S/pp_full_check.py train --tag SH30M10-F-s$s || die "training sanity SH30M10-F-s$s"; done
status "navtest / navhard (G) / HUGSIM 64 spec_plan_smooth"
"${B[@]}" run --model SH30M10-F-s0 SH30M10-F-s1 --bench navtest || die "bench navtest"
"${B[@]}" run --model SH30M10-F-s0@gimm SH30M10-F-s1@gimm --bench navhard || die "bench navhard"
"${B[@]}" run --model SH30M10-F-s0 SH30M10-F-s1 --bench hugsim --preset spec_plan_smooth --scenarios "$LIST" || die "bench hugsim"
"${B[@]}" status --model SH30M10-F-s0 SH30M10-F-s1 --bench navtest --wait || die "navtest"

status "replay and raw-vs-replay geometry"
sub m10-replay $L/replay --vram 0.5 --cpu 48 --ram 64 -- $NAV $S/turn_oracle.py replay --name m10 --models SH30M10-F-s0 SH30M10-F-s1 SH30-F-s0 SH30-F-s1 P2H10-F-s0 P2H10-F-s1
for s in 0 1; do
  PL=$("$VPY" -c "from jevdrive.bench.compat import pred_file; print(pred_file('SH30M10-F-s$s'))") || die "plans path s$s"
  sub m10-geom-s$s $L/geom-s$s --vram 0.5 --cpu 48 --ram 64 -- $NAV $S/rh.py proxy --name SH30M10-F-s$s --plans $PL --procs 48
done
waitdirs $L/replay $L/geom-s0 $L/geom-s1
sub m10-report $L/report --vram 0.5 --cpu 8 --ram 32 -- $PY $S/turn_oracle.py report --name m10 --seeds 0 1 --replays s0 b0 sh m10 \
    --arms P2H10 SH30 SH30M10 --refs SH30 P2H10 --bev SH30 SH30M10
waitdirs $L/report

status "navhard / HUGSIM reports"
"${B[@]}" status --model SH30M10-F-s0@gimm SH30M10-F-s1@gimm --bench navhard --wait || die "navhard"
"${B[@]}" status --model SH30M10-F-s0 SH30M10-F-s1 --bench hugsim --preset spec_plan_smooth --wait || die "hugsim"
"${B[@]}" report --bench navtest --arms M10=SH30M10-F-s0+SH30M10-F-s1 --vs SH30=SH30-F-s0+SH30-F-s1 P2H=P2H10-F-s0+P2H10-F-s1 --out $R || die "report navtest"
"${B[@]}" report --bench navhard --arms M10=SH30M10-F-s0@gimm+SH30M10-F-s1@gimm --vs SH30=SH30-F-s0@gimm+SH30-F-s1@gimm P2H=P2H10-F-s0@gimm+P2H10-F-s1@gimm --out $R || die "report navhard"
"${B[@]}" report --bench hugsim --preset spec_plan_smooth --arms M10=SH30M10-F-s0+SH30M10-F-s1 --vs SH30=SH30-F-s0+SH30-F-s1 P2H=P2H10-F-s0+P2H10-F-s1 \
    --scenarios all64 --out $R || die "report hugsim"
$VPY $S/rh.py geomtab --pairs SH30M10-F-s0:SH30-F-s0 SH30M10-F-s1:SH30-F-s1 SH30M10-F-s0:P2H10-F-s0 SH30M10-F-s1:P2H10-F-s1 --res-dir $R --out geom_m10 > "$R/geom.txt" || die "geomtab"
status "done"; date > "$D/DONE"
