#!/usr/bin/env bash
# One self-advancing chain of the WM-vs-reprojection check on the leased card (CUDA_VISIBLE_DEVICES set outside). STATUS / DONE / ERROR in runs/wm_vs_reproj.
set -euo pipefail
export DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}
R=$DATA_DIR/runs/wm_vs_reproj
OPPY=$DATA_DIR/envs/openpilot/bin/python
JPY=$DATA_DIR/envs/jevdrive/bin/python
S=experiments/world_model/scripts
trap 'echo "$(date +%F\ %T) failed at line $LINENO" > $R/ERROR' ERR
rm -f $R/DONE $R/ERROR
st() { echo "$(date +%F\ %T) $1" > $R/STATUS; }
st "select"; $OPPY $S/wm_prep.py select --per-cat 6 --seg-cap 6 --gap 8 --out anchors.json
mkdir -p $R/old && for d in op w report; do [ -d $R/$d ] && mv $R/$d $R/old/$d.$(date +%H%M%S) || true; done
st "openpilot branches"; $OPPY $S/wm_op.py --anchors anchors.json
st "world model"; $JPY $S/wm_w.py predict
st "report"; $JPY $S/wm_report.py
st "sheet"; $JPY $S/wm_sheet.py $R/report/wm-vs-reproj-sheet.png
st "done"; echo ok > $R/DONE
