#!/usr/bin/env bash
# op_parity hinge-scan (plans/2026-10-08-hinge-scan-prereg.md): lambda x margin grid of the drivable hinge at pilot scale + SH30 on WOD val.
# One self-advancing chain in tmux jev (scripts/tmux_run.sh hinge-scan experiments/op_parity/scripts/hinge_scan_chain.sh). Every GPU / CPU job
# through the pool, every benchmark read through jevdrive.bench.
#   WOD: SH30-F-s0 / s1 ONNX + bias -> decision 155 harness (parallel with everything else)
#   scan: 8 grid arms + SHP-F-s1 (seed noise) trained at the SHP recipe -> navtest -> replay + raw-plan geometry -> hinge_scan.py scan (winner rule)
#   full (only if the rule names a winner): SW-F-s0 / s1 on the P2H10 recipe -> navtest / navhard (G) / HUGSIM 64 -> replay, geometry -> reports vs SH30 and P2H10
# State: $DATA_DIR/runs/op_parity/hinge_scan/chain/{STATUS, DONE, ERROR, log.txt, jobs.txt, WINNER | NO_WINNER}. Rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/op_parity/hinge_scan
D=$O/chain; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
OP=$DATA_DIR/envs/openpilot/bin/python
NAV=$DATA_DIR/envs/navsim2/bin/python
J=$DATA_DIR/envs/jevdrive/bin/python
VPY=$PWD/.venv/bin/python
CL="$J -m jevdrive.cl"
S=experiments/op_parity/scripts
B=("$VPY" -m jevdrive.bench)
L=$D/pool
R=experiments/op_parity/results/hinge_scan
ONNX=$DATA_DIR/runs/op_parity/hugsim/onnx
BIAS=$DATA_DIR/runs/op_parity/wod
status() { echo "$(date '+%F %T') op_parity hinge-scan: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && return; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 20; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; return 0; }

PILOT_DATA="navtrain_full.s2of12 navtrain_full.s3of12 navtrain_full.s4of12" PILOT_SPLIT=navsim/op-parity-s234 FULL_SPLIT=navsim/op-parity-full
K=12 FULL=$(for i in $(seq 0 $((K - 1))); do echo -n "navtrain_full.s${i}of$K "; done)
LIST=experiments/hugsim/scripts/derot_all64.txt
# grid arms "tag:lambda:margin" (SHP-F-s0 = 30 / 0.5 exists; SHP-F-s1 = same setting, seed 1)
ARMS="SC-L10M25-F-s0:10:0.25:0 SC-L10M50-F-s0:10:0.5:0 SC-L10M100-F-s0:10:1.0:0 SC-L30M25-F-s0:30:0.25:0 SC-L30M100-F-s0:30:1.0:0 \
SC-L100M25-F-s0:100:0.25:0 SC-L100M50-F-s0:100:0.5:0 SC-L100M100-F-s0:100:1.0:0 SHP-F-s1:30:0.5:1"

# ---------------------------------------------------------------- WOD val: SH30 through the decision 155 harness
status "WOD: ONNX + bias + harness for SH30-F-s0 / s1"
for s in 0 1; do t=SH30-F-s$s
  [[ -f $ONNX/pp-$t.onnx ]] || $PY $S/pp_hugsim.py onnx --tag $t --out $ONNX/pp-$t.onnx > /dev/null || die "onnx $t"
  [[ -f $BIAS/bias-$t.npz ]] || $PY $S/pp_wod.py bias --tags $t || die "bias $t"
  sub hs-wod-$t $L/wod-$t --vram 8 --cpu 14 --ram 40 -- $OP scripts/wod_zeroshot_openpilot.py --set rater extra --workers 12 \
      --onnx $ONNX/pp-$t.onnx --tag $t --bias $BIAS/bias-$t.npz
done

# ---------------------------------------------------------------- scan: train, score
status "scan: train 9 pilot arms"
for arm in $ARMS; do IFS=: read -r t lam mar seed <<< "$arm"
  smoke="$PY $S/pp_train.py --arm P2 --frames warp --host --data navtrain_full.s2of12 --split $PILOT_SPLIT --steps 3 --batch 16 --eval-every 3 --hinge-lam $lam --hinge-margin $mar --tag smoke-hs-$t"
  sub hs-t-$t $L/t-$t --train --vram 24 --cpu 6 --ram 40 --preflight "$smoke" -- $PY $S/pp_train.py --arm P2 --seed $seed --frames warp --host \
      --data $PILOT_DATA --split $PILOT_SPLIT --steps 3000 --batch 64 --warmup 100 --eval-every 1000 --hinge-lam $lam --hinge-margin $mar --tag $t
done
TAGS=""
for arm in $ARMS; do IFS=: read -r t lam mar seed <<< "$arm"; TAGS="$TAGS $t"
  waitdirs $L/t-$t
  $PY $S/pp_full_check.py train --tag $t || die "training sanity $t"
  "${B[@]}" run --model $t --bench navtest || die "bench navtest $t"
done
status "scan: navtest scoring"
"${B[@]}" status --model $TAGS --bench navtest --wait || die "navtest scan"

status "scan: replay and raw-plan geometry"
sub hs-replay $L/replay --vram 0.5 --cpu 48 --ram 64 -- $NAV $S/turn_oracle.py replay --name hs --models SHP-F-s0 $TAGS
GT="SHP-F-s0 $TAGS"
for t in $GT; do
  PL=$("$VPY" -c "from jevdrive.bench.compat import pred_file; print(pred_file('$t'))") || die "plans path $t"
  [[ -f $DATA_DIR/runs/op_parity/replay_hinge/geom_$t.parquet ]] || \
    sub hs-geom-$t $L/geom-$t --vram 0.5 --cpu 24 --ram 48 -- $NAV $S/rh.py proxy --name $t --plans $PL --procs 24
done
waitdirs $L/replay; for t in $GT; do [[ -f $DATA_DIR/runs/op_parity/replay_hinge/geom_$t.parquet ]] || waitdirs $L/geom-$t; done
sub hs-scan $L/scan --vram 0.5 --cpu 8 --ram 32 -- $PY $S/hinge_scan.py scan --replays hs
waitdirs $L/scan
cp -r "$R"/verdict.json "$D/verdict.json"
BEST=$($VPY -c "import json; print(json.load(open('$D/verdict.json'))['best'] or '')")

# ---------------------------------------------------------------- WOD report (independent of the scan)
waitdirs $L/wod-SH30-F-s0 $L/wod-SH30-F-s1
$VPY $S/hinge_scan.py wod > "$D/wod_report.txt" 2>&1 || die "wod report (see $D/wod_report.txt)"

if [[ -z $BEST ]]; then
  status "scan: no winner by the pre-registered rule; SH30 stays"; touch "$D/NO_WINNER"; status "done (no winner)"; date > "$D/DONE"; exit 0
fi
# ---------------------------------------------------------------- full scale for the winner
LAM=$($VPY -c "import json; print(json.load(open('$D/verdict.json'))['arms']['$BEST']['lam'])")
MAR=$($VPY -c "import json; print(json.load(open('$D/verdict.json'))['arms']['$BEST']['margin'])")
echo "$BEST lambda $LAM margin $MAR" > "$D/WINNER"
status "full: winner $BEST (lambda $LAM, margin $MAR) at full scale, 2 seeds"
HF="--hinge-lam $LAM --hinge-margin $MAR"
smoke_f="$PY $S/pp_train.py --arm P2 --frames warp --host --data navtrain_full.s0of12 --split $FULL_SPLIT --steps 3 --batch 16 --eval-every 3 $HF --tag smoke-hs-sw"
for s in 0 1; do
  sub hs-t-SW-s$s $L/t-SW-s$s --train --vram 40 --cpu 6 --ram 40 --preflight "$smoke_f" -- $PY $S/pp_train.py --arm P2 --seed $s --frames warp --host \
      --data $FULL --split $FULL_SPLIT --steps 10000 --batch 128 --warmup 300 --eval-every 1000 $HF --tag SW-F-s$s
done
waitdirs $L/t-SW-s0 $L/t-SW-s1
for s in 0 1; do $PY $S/pp_full_check.py train --tag SW-F-s$s || die "training sanity SW-F-s$s"; done
status "full: navtest / navhard (G) / HUGSIM 64 spec_plan_smooth"
"${B[@]}" run --model SW-F-s0 SW-F-s1 --bench navtest || die "bench navtest SW"
"${B[@]}" run --model SW-F-s0@gimm SW-F-s1@gimm --bench navhard || die "bench navhard SW"
"${B[@]}" run --model SW-F-s0 SW-F-s1 --bench hugsim --preset spec_plan_smooth --scenarios "$LIST" || die "bench hugsim SW"
"${B[@]}" status --model SW-F-s0 SW-F-s1 --bench navtest --wait || die "navtest SW"
sub hs-replay-sw $L/replay-sw --vram 0.5 --cpu 48 --ram 64 -- $NAV $S/turn_oracle.py replay --name sw --models SW-F-s0 SW-F-s1 SH30-F-s0 SH30-F-s1 P2H10-F-s0 P2H10-F-s1
for s in 0 1; do
  PL=$("$VPY" -c "from jevdrive.bench.compat import pred_file; print(pred_file('SW-F-s$s'))") || die "plans path SW s$s"
  sub hs-geom-SW-s$s $L/geom-SW-s$s --vram 0.5 --cpu 48 --ram 64 -- $NAV $S/rh.py proxy --name SW-F-s$s --plans $PL --procs 48
done
waitdirs $L/replay-sw $L/geom-SW-s0 $L/geom-SW-s1
sub hs-report-sw $L/report-sw --vram 0.5 --cpu 8 --ram 32 -- $PY $S/turn_oracle.py report --name sw --seeds 0 1 --replays s0 b0 sh sw \
    --arms P2H10 SH30 SW --refs P2H10 SH30 --bev SH30 SW
waitdirs $L/report-sw
"${B[@]}" status --model SW-F-s0@gimm SW-F-s1@gimm --bench navhard --wait || die "navhard SW"
"${B[@]}" status --model SW-F-s0 SW-F-s1 --bench hugsim --preset spec_plan_smooth --wait || die "hugsim SW"
RS=$R/full
"${B[@]}" report --bench navtest --arms SW=SW-F-s0+SW-F-s1 --vs SH30=SH30-F-s0+SH30-F-s1 P2H=P2H10-F-s0+P2H10-F-s1 --out $RS || die "report navtest"
"${B[@]}" report --bench navhard --arms SW=SW-F-s0@gimm+SW-F-s1@gimm --vs SH30=SH30-F-s0@gimm+SH30-F-s1@gimm P2H=P2H10-F-s0@gimm+P2H10-F-s1@gimm --out $RS || die "report navhard"
"${B[@]}" report --bench hugsim --preset spec_plan_smooth --arms SW=SW-F-s0+SW-F-s1 --vs SH30=SH30-F-s0+SH30-F-s1 P2H=P2H10-F-s0+P2H10-F-s1 --scenarios all64 --out $RS || die "report hugsim"
$VPY $S/rh.py geomtab --pairs SW-F-s0:SH30-F-s0 SW-F-s1:SH30-F-s1 SW-F-s0:P2H10-F-s0 SW-F-s1:P2H10-F-s1 --res-dir $RS --out geom_sw > "$D/geom_sw.txt" || die "geomtab"
status "done"; date > "$D/DONE"
