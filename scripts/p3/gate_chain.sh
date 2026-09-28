#!/usr/bin/env bash
# Resolution gate chain (user 2026-09-28, (a)-(d); todos/2026-09-28-ped-dose-response.md). Box-side, unattended.
# Phase A: wait for the first donor-bank pass (scripts/p3/bank_watch.sh) -> gate check of the rendered insx items and dose
#   cells (failing ones archived) -> plan preview of all 66 insx targets under the gate -> release the insx queue -> wait
#   for the stage-2 dose driver -> re-render every stage-2 scene x state (near cells, gate-dropped cells, collision-course
#   cells) -> counts. Marker A_DONE.
# Phase B: the registered readout chain with the threat split -> readout_final/, then the R3D2 check. Marker DONE.
# Any failure writes ERROR (with the step) and stops.   scripts/tmux_run.sh p3-gate bash scripts/p3/gate_chain.sh
set -euo pipefail
cd ~/data/jev-drive
P3=$DATA_DIR/runs/nq4/p3; X=$P3/xinsert; DO=$P3/dose; G=$DO/gate; mkdir -p "$G"
PY=$DATA_DIR/envs/drivestudio/bin/python; JV=$DATA_DIR/envs/jevdrive/bin/python; OP=$DATA_DIR/envs/openpilot/bin/python
RP=$DATA_DIR/envs/r3d2/bin/python
L=$G/log.txt
log() { echo "$(date '+%F %T') $*" | tee -a "$L"; }
trap 'log "FAILED at line $LINENO"; echo "line $LINENO" > $G/ERROR' ERR
rm -f "$G/ERROR"
export TORCH_EXTENSIONS_DIR=$DATA_DIR/cache/torch_ext_p3x P3_PRELOAD_DEVICE=cpu

pick_gpu() {   # a shared card with >= 30 GB free; GPU 6 only while it has >= 55 GB free (our share there is <= 25 GB)
  while true; do
    for g in 2 3 4 5 6; do
      free=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
      need=30000; [ "$g" = 6 ] && need=55000
      [ -n "$free" ] && [ "$free" -ge $need ] && { echo "$g"; return; }
    done
    sleep 60
  done
}

if [ ! -f "$G/A_DONE" ]; then
  log "phase A: waiting for the first bank pass"
  until [ -f "$X/bank/PASS1_DONE" ]; do sleep 120; done
  log "bank: $(cat $X/bank/summary.json)"
  cd scripts/p3
  taskset -c 176,177 $PY xinsert.py gatecheck --replan-skipped 2>&1 | tail -3 | tee -a "$L"
  # plan preview of all targets under the gate (counts only; the lane re-plans into plan/ itself)
  rm -rf "$X/plan_gate_preview"
  for s in 0 1 2 3; do
    taskset -c $((176 + s % 2)) $PY xinsert.py plan --plan-dir "$X/plan_gate_preview" --targets $(seq $s 4 65) > "$G/preview_$s.out" 2>&1 &
  done
  wait
  cd ~/data/jev-drive
  # release the insx queue (only the holds this chain's operator placed)
  grep -l "resolution gate + donor bank expansion" "$P3"/expand/skip/ins_* 2>/dev/null | xargs -r rm -f
  log "insx queue released"
  log "waiting for the stage-2 dose driver"
  until [ -f "$DO/STAGE2_DONE" ] || [ -f "$DO/STAGE2_ERROR" ]; do sleep 120; done
  (cd scripts/p3 && taskset -c 176,177 $PY dose.py gatecheck 2>&1 | tail -3 | tee -a "$L")
  rm -f "$DO/NEAR_HOLD"
  mapfile -t units < <($JV -c "
import json; a = json.load(open('$DO/anchors.json'))
print('\n'.join(f'{k}:{s}' for k, v in a.items() if int(k) <= 10 for s in v))")
  log "re-render ${#units[@]} scene x state units: ${units[*]}"
  i=0
  for u in "${units[@]}"; do
    k=${u%%:*}; st=${u##*:}; g=$(pick_gpu)
    cores=$((160 + 2 * (i % 3))),$((161 + 2 * (i % 3)))
    log "render $u on GPU $g"
    ( CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 taskset -c $cores timeout 14400 $PY scripts/p3/dose.py render --scene "$k" --state "$st" \
        > "$G/render_${k}_${st}.out" 2>&1 || echo "$u" >> "$G/render_failed.txt" ) &
    i=$((i + 1))
    [ $((i % 3)) -eq 0 ] && wait
    sleep 90                                              # let the job allocate before the next free-VRAM reading
  done
  wait
  [ -s "$G/render_failed.txt" ] && { log "render failures: $(cat $G/render_failed.txt)"; echo "render" > "$G/ERROR"; exit 1; }
  $JV - > "$G/counts.json" <<EOF
import json, glob, os
from pathlib import Path
DO = Path("$DO"); X = Path("$X")
c = {"cells": 0, "near_cells": 0, "collide_cells": 0, "skipped": {}, "gate_dropped": len(glob.glob(str(DO / "items/*/_gate_dropped/*")))}
for m in DO.glob("items/p3_*_*/*/meta.json"):
    q = json.loads(m.read_text()); c["cells"] += 1; c["near_cells"] += q["dist"] < 20; c["collide_cells"] += q["ped_state"] == "collide"
for s in DO.glob("items/p3_*_*/*/skipped.json"):
    r = json.loads(s.read_text())["reason"].split(":")[0].split(" (")[0]; c["skipped"][r] = c["skipped"].get(r, 0) + 1
pv = [json.loads(p.read_text()) for p in (X / "plan_gate_preview").glob("p3_*.json")]
c["insx_preview"] = {"targets": len(pv), "items": sum(bool(p.get("item")) for p in pv)}
c["insx_dropped"] = len(list((X / "items_dropped_gate").glob("p3_*")))
c["bank"] = json.loads((X / "bank/summary.json").read_text())
print(json.dumps(c, indent=1))
EOF
  log "counts: $(tr -d '\n ' < $G/counts.json)"
  touch "$G/A_DONE"
fi

log "phase B: readout"
export P3_SET=nq4_p3_dose P5_SET=nq4_p3_dose CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 P3_HEAD_TOL=0.5
rm -rf "$DATA_DIR/processed/nq4_p3_dose"
taskset -c 168,169 $PY scripts/p3/dose.py stage | tee -a "$L"
taskset -c 168,169 $JV -m jevdrive.nq4_p3 index --processed-root $DATA_DIR/processed/waymo_ds/training >> "$L" 2>&1
taskset -c 168,169 $OP scripts/p5_openpilot.py --models cinque lebowski --workers 2 --arrays temporal plan lead lead_prob >> "$G/openpilot.out" 2>&1
$JV -m jevdrive.p5_openpilot finalize --models cinque,lebowski --arrays temporal,plan,lead,lead_prob >> "$L" 2>&1
taskset -c 168,169 $JV -m jevdrive.nq4_p3 exam > "$G/exam.out" 2>&1
grep -q "prior does not reproduce" "$G/exam.out" && { log "head tol not applied"; echo exam > "$G/ERROR"; exit 1; }
EX=$(ls -td $DATA_DIR/runs/nq4/nq4_p3_dose-exam/*/ | head -1)
taskset -c 168,169 $JV -m jevdrive.ped_dose --exam-dir "$EX" --out "$DO/readout_final" > "$G/ped_dose.out" 2>&1
log "readout done: $DO/readout_final"
touch "$G/READOUT_DONE"

log "R3D2 check"
CELLS=$DO/readout_final/cells.csv
$JV scripts/p3/dose_r3d2.py select --cells "$CELLS" | tee -a "$L"
until [ "$(nvidia-smi -i 6 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 30000 ]; do sleep 60; done
CUDA_VISIBLE_DEVICES=6 taskset -c 168,169 $RP scripts/p3/dose_r3d2.py render > "$G/r3d2_render.out" 2>&1
export P3_SET=nq4_p3_dose_r3d2 P5_SET=nq4_p3_dose_r3d2
rm -rf "$DATA_DIR/processed/nq4_p3_dose_r3d2"
$JV scripts/p3/dose_r3d2.py stage | tee -a "$L"
taskset -c 168,169 $JV -m jevdrive.nq4_p3 index --processed-root $DATA_DIR/processed/waymo_ds/training >> "$L" 2>&1
taskset -c 168,169 $OP scripts/p5_openpilot.py --models cinque lebowski --workers 2 --arrays temporal plan lead lead_prob >> "$G/openpilot_r3d2.out" 2>&1
$JV -m jevdrive.p5_openpilot finalize --models cinque,lebowski --arrays temporal,plan,lead,lead_prob >> "$L" 2>&1
taskset -c 168,169 $JV -m jevdrive.nq4_p3 exam > "$G/exam_r3d2.out" 2>&1
EX=$(ls -td $DATA_DIR/runs/nq4/nq4_p3_dose_r3d2-exam/*/ | head -1)
taskset -c 168,169 $JV -m jevdrive.ped_dose --exam-dir "$EX" --out "$DO/r3d2/readout" > "$G/ped_dose_r3d2.out" 2>&1
$JV scripts/p3/dose_r3d2.py compare --cells "$CELLS" | tee -a "$L"
touch "$G/DONE"
log "all done"
