#!/usr/bin/env bash
# Lane WOD1 (plans/2026-10-09-wod1-recipes-prereg.md): serve the 2026-10-09 recipes on WOD-E2E val through the decision 155 harness, then report.
# Run: scripts/tmux_run.sh wod1 experiments/op_parity/scripts/wod1_recipes_chain.sh
# State: $DATA_DIR/runs/op_parity/wod1/{STATUS, DONE, ERROR, log.txt, pool/}. Rerunning resumes. At most 3 WOD eval jobs at a time.
set -uo pipefail
cd "$(dirname "$0")/../../.."
D=$DATA_DIR/runs/op_parity/wod1; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
OP=$DATA_DIR/envs/openpilot/bin/python
J=$DATA_DIR/envs/jevdrive/bin/python
CL="$J -m jevdrive.cl"
S=experiments/op_parity/scripts
L=$D/pool
ONNX=$DATA_DIR/runs/op_parity/hugsim/onnx
BIAS=$DATA_DIR/runs/op_parity/wod
status() { echo "$(date '+%F %T') wod1: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && return; rm -f "$ld/ERROR"
        $CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@" >/dev/null || die "submit $n"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 20; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; return 0; }
serve() { local tag=$1
          [[ -f $ONNX/pp-$tag.onnx ]] || $PY $S/pp_hugsim.py onnx --tag $tag --out $ONNX/pp-$tag.onnx >/dev/null || die "onnx $tag"
          [[ -f $BIAS/bias-$tag.npz ]] || $PY $S/pp_wod.py bias --tags $tag || die "bias $tag"
          sub wod1-e-$tag $L/e-$tag --vram 8 --cpu 14 --ram 40 -- $OP scripts/wod_zeroshot_openpilot.py --set rater extra --workers 12 \
              --onnx $ONNX/pp-$tag.onnx --tag $tag --bias $BIAS/bias-$tag.npz; }
# restore check: sets.json was missing on the box and was rebuilt from the index; re-serve stored P2H10-F-s0 under tag chk-P2H10-F-s0, must equal the stored predictions
serve_chk() { sub wod1-e-chk $L/e-chk --vram 8 --cpu 14 --ram 40 -- $OP scripts/wod_zeroshot_openpilot.py --set rater extra --workers 12 \
              --onnx $ONNX/pp-P2H10-F-s0.onnx --tag chk-P2H10-F-s0 --bias $BIAS/bias-P2H10-F-s0.npz; }
wave() { status "serving wave: $*"; for t in "$@"; do serve $t; done; waitdirs $(for t in "$@"; do echo $L/e-$t; done); }
serve_chk; serve P2-F-s0; serve P2-F-s1; waitdirs $L/e-chk $L/e-P2-F-s0 $L/e-P2-F-s1
wave OT10a05-F-s0 OT10a05-F-s1 YR10m10-F-s0
wave YR10m10-F-s1 YR10m25-F-s0 YR10m25-F-s1
status "report"
R=experiments/op_parity/results/wod1_recipes
$J $S/wod_slot.py report --out $R --arms shipped P2H10=P2H10-F-s0+P2H10-F-s1 P2=P2-F-s0+P2-F-s1 SH30=SH30-F-s0+SH30-F-s1 \
    OT10a05=OT10a05-F-s0+OT10a05-F-s1 YR10m10=YR10m10-F-s0+YR10m10-F-s1 YR10m25=YR10m25-F-s0+YR10m25-F-s1 \
    --pairs P2:P2H10 SH30:P2H10 OT10a05:P2H10 YR10m10:P2H10 YR10m25:P2H10 P2H10:shipped || die "report"
status "done"; date > "$D/DONE"
