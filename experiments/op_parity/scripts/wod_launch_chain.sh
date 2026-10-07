#!/usr/bin/env bash
# op_parity / wod-launch lane (plans/2026-10-08-wod-launch-prereg.md): why WP2 under-launches from standstill on WOD val, then the recipe fix.
# One self-advancing chain in tmux jev (scripts/tmux_run.sh); every GPU job goes through the pool. Rerunning resumes.
#   diag                 Step 1: VLM context labels, ego-channel bias arms through the harness (rater frames), the token-path arms
#   fix <ARM> <flags..>  Step 2: pilot of the WOD recipe with the extra pp_train flags (tag <ARM>-pilot-s0) -> gate G-L -> full, 2 seeds
# State: $DATA_DIR/runs/op_parity/wod/launch/{STATUS, DONE-<stage>, ERROR, GATE_STOP, chain.log, jobs.txt}.
set -uo pipefail
cd "$(dirname "$0")/../../.."
STAGE=${1:?stage: diag | fix}; shift
D=$DATA_DIR/runs/op_parity/wod/launch; mkdir -p "$D"; rm -f "$D/DONE-$STAGE" "$D/ERROR" "$D/GATE_STOP"
exec > >(tee -a "$D/chain.log") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
OP=$DATA_DIR/envs/openpilot/bin/python
J=$DATA_DIR/envs/jevdrive/bin/python
CL="$J -m jevdrive.cl"
S=experiments/op_parity/scripts
L=$D/pool
ONNX=$DATA_DIR/runs/op_parity/hugsim/onnx
BIAS=$DATA_DIR/runs/op_parity/wod
VARS="main zero biasmean biasresid cmd0 acc0 vx1 vx3 cv1 cv3"
status() { echo "$(date '+%F %T') op_parity wod_launch: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && { echo done; return; }
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && { echo "$live"; return; }; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; echo "$id"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 30; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; }

if [[ $STAGE == diag ]]; then
  status "diag: bias files"
  for t in WP2-full-s0 WP2-full-s1; do [[ -f $D/bias-${t}_cv3.npz ]] || $PY $S/wod_launch.py bias --tags $t || die "bias $t"; done
  status "diag: submit label / tok / serve"
  sub wodl-label $L/label --vram 14 --cpu 6 --ram 24 -- $J $S/wod_launch.py label >/dev/null
  sub wodl-tok $L/tok --vram 16 --cpu 8 --ram 48 -- $PY $S/wod_launch.py tok >/dev/null
  for t in WP2-full-s0 WP2-full-s1; do
    tags=""; bs=""; for v in $VARS; do tags="$tags lx-${t}_$v"; bs="$bs $D/bias-${t}_$v.npz"; done
    sub wodl-e-$t $L/e-$t --vram 8 --cpu 14 --ram 40 -- $OP scripts/wod_zeroshot_openpilot.py --set rater --workers 12 \
        --onnx $ONNX/pp-$t.onnx --tag $tags --bias $bs >/dev/null
  done
  waitdirs $L/label $L/tok $L/e-WP2-full-s0 $L/e-WP2-full-s1
  status "diag: jobs done"; touch "$D/DONE-diag"; exit 0
fi

if [[ $STAGE == fix ]]; then
  ARM=${1:?arm name}; shift; FLAGS="$*"
  train() { local tag=$1 seed=$2 data=$3; shift 3
            sub wodl-t-$tag $L/t-$tag --train --vram 24 --cpu 6 --ram 40 -- $PY $S/pp_train.py --arm P2 --seed $seed --host --data $data \
                --split wod/r2 --tag $tag $FLAGS "$@" >/dev/null; }
  serve() { local tag=$1
            [[ -f $ONNX/pp-$tag.onnx ]] || $PY $S/pp_hugsim.py onnx --tag $tag --out $ONNX/pp-$tag.onnx >/dev/null || die "onnx $tag"
            [[ -f $BIAS/bias-$tag.npz ]] || { if [[ "$FLAGS" == *--stop-gate* ]]; then $PY $S/wod_launch.py gbias --tags $tag; else $PY $S/pp_wod.py bias --tags $tag; fi; } || die "bias $tag"
            sub wodl-e-$tag $L/e-$tag --vram 8 --cpu 14 --ram 40 -- $OP scripts/wod_zeroshot_openpilot.py --set rater extra --workers 12 \
                --onnx $ONNX/pp-$tag.onnx --tag $tag --bias $BIAS/bias-$tag.npz >/dev/null; }
  status "fix $ARM: pilot train ($FLAGS)"
  train $ARM-pilot-s0 0 wod_pilot --steps 600 --batch 64 --warmup 100 --eval-every 200
  waitdirs $L/t-$ARM-pilot-s0
  status "fix $ARM: pilot serve"
  serve $ARM-pilot-s0; waitdirs $L/e-$ARM-pilot-s0
  $J $S/wod_launch_report.py gate --arm $ARM-pilot-s0 --ref WP2-pilot-s0
  rc=$?
  if [[ $rc == 3 ]]; then status "STOP: pilot gate G-L failed for $ARM ($(cat experiments/op_parity/results/wod_launch/gate_$ARM.json))"; touch "$D/GATE_STOP"; exit 0; fi
  [[ $rc == 0 ]] || die "pilot gate report"
  status "fix $ARM: gate passed ($(cat experiments/op_parity/results/wod_launch/gate_$ARM.json)); full, 2 seeds"
  for s in 0 1; do train $ARM-full-s$s $s wod_r2 --steps 10000 --batch 128 --warmup 300 --eval-every 1000; done
  for s in 0 1; do waitdirs $L/t-$ARM-full-s$s; serve $ARM-full-s$s; done
  waitdirs $L/e-$ARM-full-s0 $L/e-$ARM-full-s1
  $J $S/wod_launch_report.py full --arm $ARM || die "full report"
  status "fix $ARM: done"; touch "$D/DONE-fix"; exit 0
fi
die "unknown stage $STAGE"
