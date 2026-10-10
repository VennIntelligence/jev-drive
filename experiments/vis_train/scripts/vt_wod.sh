#!/usr/bin/env bash
# vis_train: zero-shot WOD-E2E val of final P2-format checkpoints (F, F0, and C exported as a P2 tag), through the op_parity pipeline of
# wod_parity_chain.sh `serve`: ONNX (pp_hugsim.py onnx) -> ego bias (pp_wod.py bias) -> openpilot harness (wod_zeroshot_openpilot.py, rater + extra
# sets) -> preds/op_cinque_<tag>. Each tag is three chained pool jobs, the first gated on the checkpoint, all low priority. Resubmitting skips a
# tag whose job (same name) is queued or running. Memory-branch arms (A0 / A / B / W) need the branch encoder in the harness: not run.
#   scripts/vt_wod.sh VT-F-s0 VT-F-s1 VT-F0-s0 VTCP2-s0        (box, repo root; VTCP2-s0 = vt_native.py export of VT-C-s0)
# Read afterwards (reference rows SH30-F-s{0,1} and shipped are stored):
#   python experiments/op_parity/scripts/wod_parity.py report --name vt --arms SH30=SH30-F-s0+SH30-F-s1 F=VT-F-s0+VT-F-s1 F0=VT-F0-s0 C=VTCP2-s0 \
#       --pairs F:SH30 F0:F C:F0 C:shipped        (jevdrive env)
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/vis_train/wod; ONNX=$DATA_DIR/runs/op_parity/hugsim/onnx; BIAS=$DATA_DIR/runs/op_parity/wod
S=experiments/op_parity/scripts
PY=$DATA_DIR/envs/op-train/bin/python OP=$DATA_DIR/envs/openpilot/bin/python
CL=".venv/bin/python -m jevdrive.cl"
live() { $CL queue 2>/dev/null | awk -v n="$1" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}'; }
for tag in "$@"; do
  ck=$DATA_DIR/runs/op_parity/runs/$tag/ckpt-final.pt
  n1=vt-wod-onnx-$tag n2=vt-wod-bias-$tag n3=vt-wod-run-$tag
  [[ -n $(live $n1)$(live $n2)$(live $n3) ]] && { echo "$tag: already queued"; continue; }
  mkdir -p "$O/$tag"
  j1=$($CL submit --owner vis_train --name $n1 --log-dir $O/$tag/onnx --vram 0.5 --cpu 4 --ram 24 --priority 2 --when-exists "$ck" -- \
       bash -c "[[ -f $ONNX/pp-$tag.onnx ]] || $PY $S/pp_hugsim.py onnx --tag $tag --out $ONNX/pp-$tag.onnx")
  j2=$($CL submit --owner vis_train --name $n2 --log-dir $O/$tag/bias --vram 0.5 --cpu 4 --ram 24 --priority 2 --after "$j1" -- \
       bash -c "[[ -f $BIAS/bias-$tag.npz ]] || $PY $S/pp_wod.py bias --tags $tag")
  j3=$($CL submit --owner vis_train --name $n3 --log-dir $O/$tag/run --vram 8 --cpu 14 --ram 40 --priority 2 --after "$j2" -- \
       $OP scripts/wod_zeroshot_openpilot.py --set rater extra --workers 12 --onnx $ONNX/pp-$tag.onnx --tag $tag --bias $BIAS/bias-$tag.npz)
  echo "$tag: $j1 $j2 $j3" | tee -a $O/jobs.txt
done
