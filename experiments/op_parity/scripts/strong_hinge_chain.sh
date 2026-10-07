#!/usr/bin/env bash
# op_parity strong-hinge (plans/2026-10-08-strong-hinge-prereg.md): P2 + footprint drivable hinge at lambda 30 / margin 0.5 m, no memory tokens.
# One self-advancing chain in tmux jev (scripts/tmux_run.sh strong-hinge experiments/op_parity/scripts/strong_hinge_chain.sh). Every GPU / CPU
# job through the pool, every benchmark read through jevdrive.bench.
#   pilot SHP-F-s0 (H0 recipe, 3 000 steps) -> navtest -> GATE (EPDMS gain >= +0.3 and straight EPDMS >= -0.2 vs RH0-F-s0; else GATE_STOP)
#   -> SH30-F-s0 / s1 (P2H10 recipe) -> navtest / navhard (G) / HUGSIM 64 spec_plan_smooth -> four_dirs replay + raw-vs-replay geometry -> reports
# State: $DATA_DIR/runs/op_parity/strong_hinge/chain/{STATUS, DONE, ERROR, GATE_STOP, log.txt, jobs.txt}; results under .../results/. Rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/op_parity/strong_hinge
D=$O/chain; mkdir -p "$D" "$O/results"; rm -f "$D/DONE" "$D/ERROR" "$D/GATE_STOP"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
NAV=$DATA_DIR/envs/navsim2/bin/python
VPY=$PWD/.venv/bin/python
CL="$DATA_DIR/envs/jevdrive/bin/python -m jevdrive.cl"
S=experiments/op_parity/scripts
B=("$VPY" -m jevdrive.bench)
L=$D/pool
R=$O/results
HF="--hinge-lam 30 --hinge-margin 0.5"
K=12 FULL=$(for i in $(seq 0 $((K - 1))); do echo -n "navtrain_full.s${i}of$K "; done)
PILOT_DATA="navtrain_full.s2of12 navtrain_full.s3of12 navtrain_full.s4of12" PILOT_SPLIT=navsim/op-parity-s234 FULL_SPLIT=navsim/op-parity-full
LIST=experiments/hugsim/scripts/derot_all64.txt
status() { echo "$(date '+%F %T') op_parity strong-hinge: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && return; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 20; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; return 0; }

# ---------------------------------------------------------------- 1. pilot gate (seed 0, H0 recipe, no memory)
smoke_p="$PY $S/pp_train.py --arm P2 --frames warp --host --data navtrain_full.s2of12 --split $PILOT_SPLIT --steps 3 --batch 16 --eval-every 3 $HF --tag smoke-sh"
status "pilot: train SHP-F-s0"
sub sh-t-SHP-s0 $L/t-SHP-s0 --train --vram 24 --cpu 6 --ram 40 --preflight "$smoke_p" -- $PY $S/pp_train.py --arm P2 --seed 0 --frames warp --host \
    --data $PILOT_DATA --split $PILOT_SPLIT --steps 3000 --batch 64 --warmup 100 --eval-every 1000 $HF --tag SHP-F-s0
waitdirs $L/t-SHP-s0
$PY $S/pp_full_check.py train --tag SHP-F-s0 || die "training sanity SHP-F-s0"
status "pilot: navtest"
"${B[@]}" run --model SHP-F-s0 --bench navtest --wait || die "bench navtest SHP-F-s0"
$VPY $S/strong_hinge.py gate --new SHP-F-s0 --ref RH0-F-s0 > "$D/gate.txt" 2>&1; rc=$?
tail -n 12 "$D/gate.txt"
(( rc == 0 || rc == 2 )) || die "gate"
pilot_report() {
  sub sh-replay-pilot $L/replay-pilot --vram 0.5 --cpu 48 --ram 64 -- $NAV $S/turn_oracle.py replay --name shp --models SHP-F-s0
  waitdirs $L/replay-pilot
  sub sh-report-pilot $L/report-pilot --vram 0.5 --cpu 8 --ram 32 -- $PY $S/turn_oracle.py report --name sh_pilot --seeds 0 --replays s0 b0 shp \
      --arms H0 OS OSh SHP --refs H0 --bev H0 SHP
  waitdirs $L/report-pilot
}
if (( rc == 2 )); then
  status "pilot gate: STOP (EPDMS gain < +0.3 or straight EPDMS drop > 0.2)"; cp "$O/gate.json" "$D/GATE_STOP"
  pilot_report; status "done (gate stop)"; date > "$D/DONE"; exit 0
fi
status "pilot gate passed; launching the full recipe"

# ---------------------------------------------------------------- 2. full P2H10 recipe x 2 seeds
smoke_f="$PY $S/pp_train.py --arm P2 --frames warp --host --data navtrain_full.s0of12 --split $FULL_SPLIT --steps 3 --batch 16 --eval-every 3 $HF --tag smoke-sh30"
for s in 0 1; do
  sub sh-t-SH30-s$s $L/t-SH30-s$s --train --vram 40 --cpu 6 --ram 40 --preflight "$smoke_f" -- $PY $S/pp_train.py --arm P2 --seed $s --frames warp --host \
      --data $FULL --split $FULL_SPLIT --steps 10000 --batch 128 --warmup 300 --eval-every 1000 $HF --tag SH30-F-s$s
done
waitdirs $L/t-SH30-s0 $L/t-SH30-s1
for s in 0 1; do $PY $S/pp_full_check.py train --tag SH30-F-s$s || die "training sanity SH30-F-s$s"; done
status "full: navtest / navhard (G) / HUGSIM 64 spec_plan_smooth"
"${B[@]}" run --model SH30-F-s0 SH30-F-s1 --bench navtest || die "bench navtest"
"${B[@]}" run --model SH30-F-s0@gimm SH30-F-s1@gimm --bench navhard || die "bench navhard"
"${B[@]}" run --model SH30-F-s0 SH30-F-s1 --bench hugsim --preset spec_plan_smooth --scenarios "$LIST" || die "bench hugsim"
"${B[@]}" status --model SH30-F-s0 SH30-F-s1 --bench navtest --wait || die "navtest"

status "full: four_dirs replay and raw-vs-replay geometry"
sub sh-replay $L/replay --vram 0.5 --cpu 48 --ram 64 -- $NAV $S/turn_oracle.py replay --name sh \
    --models SH30-F-s0 SH30-F-s1 SHP-F-s0 P2H10-F-s0 P2H10-F-s1 RMH10-F-s0 RMH10-F-s1
waitdirs $L/replay
for s in 0 1; do
  PL=$("$VPY" -c "from jevdrive.bench.compat import pred_file; print(pred_file('SH30-F-s$s'))") || die "plans path s$s"
  sub sh-geom-s$s $L/geom-s$s --vram 0.5 --cpu 48 --ram 64 -- $NAV $S/rh.py proxy --name SH30-F-s$s --plans $PL --procs 48
done
waitdirs $L/geom-s0 $L/geom-s1
sub sh-report $L/report --vram 0.5 --cpu 8 --ram 32 -- $PY $S/turn_oracle.py report --name sh --seeds 0 1 --replays s0 b0 sh \
    --arms P2H10 RMH10 SH30 --refs P2H10 RMH10 --bev P2H10 SH30
sub sh-report-pilot $L/report-pilot --vram 0.5 --cpu 8 --ram 32 -- $PY $S/turn_oracle.py report --name sh_pilot --seeds 0 --replays s0 b0 sh \
    --arms H0 OS OSh SHP --refs H0 --bev H0 SHP
waitdirs $L/report $L/report-pilot

status "full: navtest / navhard / HUGSIM reports"
"${B[@]}" status --model SH30-F-s0@gimm SH30-F-s1@gimm --bench navhard --wait || die "navhard"
"${B[@]}" status --model SH30-F-s0 SH30-F-s1 --bench hugsim --preset spec_plan_smooth --wait || die "hugsim"
for h in P2H10; do for s in 0 1; do g=$DATA_DIR/runs/op_parity/navhard_gimm/harness/$h-F-s$s; [[ -e $g ]] || ln -s $DATA_DIR/runs/op_parity/hinge/harness/$h-F-s$s $g; done; done
"${B[@]}" report --bench navtest --arms SH=SH30-F-s0+SH30-F-s1 --vs P2H=P2H10-F-s0+P2H10-F-s1 RMH=RMH10-F-s0+RMH10-F-s1 WA-JEPA --out $R || die "report navtest"
"${B[@]}" report --bench navhard --arms SH=SH30-F-s0@gimm+SH30-F-s1@gimm --vs P2H=P2H10-F-s0@gimm+P2H10-F-s1@gimm RMH=RMH10-F-s0@gimm+RMH10-F-s1@gimm WA-JEPA \
    --out $R || die "report navhard"
"${B[@]}" report --bench hugsim --preset spec_plan_smooth --arms SH=SH30-F-s0+SH30-F-s1 --vs P2H=P2H10-F-s0+P2H10-F-s1 RMH=RMH10-F-s0+RMH10-F-s1 \
    --scenarios all64 --out $R || die "report hugsim"
$VPY $S/rh.py geomtab --pairs SH30-F-s0:P2H10-F-s0 SH30-F-s1:P2H10-F-s1 SH30-F-s0:RMH10-F-s0 SH30-F-s1:RMH10-F-s1 RMH10-F-s0:P2H10-F-s0 \
    --res-dir $R --out geom_sh > "$R/geom.txt" || die "geomtab"
status "done"; date > "$D/DONE"
