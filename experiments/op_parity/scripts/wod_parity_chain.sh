#!/usr/bin/env bash
# op_parity WOD parity lane (plans/2026-10-07-wod-parity-prereg.md): the P2 recipe trained on WOD r2-train, read on WOD val through the
# existing harness. One self-advancing chain in tmux jev (scripts/tmux_run.sh); every GPU job goes through the pool.
#   0. (done before) wod_parity.py prep -> cache/wod_r2 + cache/wod_pilot; wod_parity.py check (G0, runs/op_parity/wod/check.json)
#   1. pilot: WP2-pilot-s0 / WP1-pilot-s0 (600 steps, batch 64) -> ONNX + bias -> harness -> report pilot; gate G1: d RFS(WP2 - shipped) >= -0.10
#   2. full: WP2-full-s0/s1, WP1-full-s0/s1 (10 000 steps, batch 128) -> ONNX + bias -> harness -> report full
# State: $DATA_DIR/runs/op_parity/wod/parity/{STATUS, DONE, ERROR, GATE_STOP, chain.log, jobs.txt}. Rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
D=$DATA_DIR/runs/op_parity/wod/parity; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR" "$D/GATE_STOP"
exec > >(tee -a "$D/chain.log") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
OP=$DATA_DIR/envs/openpilot/bin/python
J=$DATA_DIR/envs/jevdrive/bin/python
CL="$J -m jevdrive.cl"
S=experiments/op_parity/scripts
L=$D/pool
ONNX=$DATA_DIR/runs/op_parity/hugsim/onnx
BIAS=$DATA_DIR/runs/op_parity/wod
P2H="P2H=P2H10-F-s0+P2H10-F-s1"
status() { echo "$(date '+%F %T') op_parity wod_parity: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && { echo done; return; }
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && { echo "$live"; return; }; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; echo "$id"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 30; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; }
train() { local tag=$1 arm=$2 seed=$3 data=$4; shift 4
          sub wodp-t-$tag $L/t-$tag --train --vram 24 --cpu 6 --ram 40 -- $PY $S/pp_train.py --arm $arm --seed $seed --host --data $data \
              --split wod/r2 --tag $tag "$@" >/dev/null; }
serve() { local tag=$1
          [[ -f $ONNX/pp-$tag.onnx ]] || $PY $S/pp_hugsim.py onnx --tag $tag --out $ONNX/pp-$tag.onnx >/dev/null || die "onnx $tag"
          [[ -f $BIAS/bias-$tag.npz ]] || $PY $S/pp_wod.py bias --tags $tag || die "bias $tag"
          sub wodp-e-$tag $L/e-$tag --vram 8 --cpu 14 --ram 40 -- $OP scripts/wod_zeroshot_openpilot.py --set rater extra --workers 12 \
              --onnx $ONNX/pp-$tag.onnx --tag $tag --bias $BIAS/bias-$tag.npz >/dev/null; }

python3 -c "import json,sys; d=json.load(open('$BIAS/check.json')); sys.exit(0 if d['gate_mean_lt_0p05'] else 1)" || die "G0 equivalence check"

# ---------------------------------------------------------------- 1. pilot
status "pilot: train WP2-pilot-s0 / WP1-pilot-s0"
PIL="--steps 600 --batch 64 --warmup 100 --eval-every 200"
train WP2-pilot-s0 P2 0 wod_pilot $PIL
train WP1-pilot-s0 P1 0 wod_pilot $PIL
waitdirs $L/t-WP2-pilot-s0 $L/t-WP1-pilot-s0
status "pilot: serve on WOD val"
for t in WP2-pilot-s0 WP1-pilot-s0; do serve $t; done
waitdirs $L/e-WP2-pilot-s0 $L/e-WP1-pilot-s0
$J $S/wod_parity.py report --name pilot --arms WP2=WP2-pilot-s0 WP1=WP1-pilot-s0 $P2H --pairs WP2:shipped WP1:shipped WP2:WP1 WP2:P2H \
    --gate -0.10
rc=$?
if [[ $rc == 3 ]]; then status "STOP: pilot gate G1 failed ($(cat experiments/op_parity/results/wod_parity/pilot_gate.json))"; touch "$D/GATE_STOP"; exit 0; fi
[[ $rc == 0 ]] || die "pilot report"
status "pilot gate passed: $(cat experiments/op_parity/results/wod_parity/pilot_gate.json)"

# ---------------------------------------------------------------- 2. full, 2 seeds
status "full: train WP2 / WP1 x seeds 0, 1"
FUL="--steps 10000 --batch 128 --warmup 300 --eval-every 1000"
for s in 0 1; do train WP2-full-s$s P2 $s wod_r2 $FUL; train WP1-full-s$s P1 $s wod_r2 $FUL; done
for s in 0 1; do for a in WP2 WP1; do waitdirs $L/t-$a-full-s$s; serve $a-full-s$s; done; done
waitdirs $L/e-WP2-full-s0 $L/e-WP1-full-s0 $L/e-WP2-full-s1 $L/e-WP1-full-s1
$J $S/wod_parity.py report --name full --arms WP2=WP2-full-s0+WP2-full-s1 WP1=WP1-full-s0+WP1-full-s1 $P2H \
    --pairs WP2:shipped WP1:shipped WP2:WP1 WP2:P2H P2H:shipped || die "full report"
status "done"; touch "$D/DONE"
