#!/usr/bin/env bash
# Night queue 3, lane C: the step bodies of scripts/nq3_c.sh (sourced; each step runs as `bash -c "source ...; <step>"`
# under the chain's timeout). Environment shared by every step.
: "${DATA_DIR:?}"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
R=$DATA_DIR/runs/nq3/c
PY=$REPO/.venv/bin/python
OPPY=$DATA_DIR/envs/openpilot/bin/python
ULPY=$DATA_DIR/envs/ultralytics/bin/python
export CUDA_VISIBLE_DEVICES=6 P6=carla_p6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

retry() {   # retry <tries> <cmd...>: GPU 6 is oversubscribed by five lanes; an OOM at load waits 2 min and tries again
  local n=$1 i; shift
  for ((i = 1; i <= n; i++)); do "$@" && return 0; echo "attempt $i of $n failed: $*"; sleep 120; done
  return 1
}

feats() {   # C1: Qwen (cores 164-179) || V-JEPA 2 -> YOLO detect -> tokens (cores 172-179)
  local pids=() rc=0
  # CPU-bound (HF processor): one process (~10 GB of VRAM on a full card) with 12 loader workers on the 16 cores;
  # retried for up to ~7 h while GPU 6 has no room to load it
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 retry 200 taskset -c 164-179 \
    $PY -m jevdrive.nq3_feats qwen --shard 0/1 --batch 2 --workers 12 > "$R/feats/qwen0.log" 2>&1 & pids+=($!)
  (
    set -e
    [[ -f $DATA_DIR/processed/$P6/bb_vjepa2/mean.npy ]] || \
      OMP_NUM_THREADS=2 retry 20 taskset -c 172-179 $PY -m jevdrive.nq3_feats vjepa --batch 64 --workers 6
    for i in 0 1; do
      OMP_NUM_THREADS=1 retry 20 taskset -c $((172 + 4 * i))-$((175 + 4 * i)) $ULPY -m jevdrive.night2_n4 detect --part $i/2 \
        --images "$DATA_DIR/processed/$P6/nq3_images.parquet" --dst "$DATA_DIR/processed/$P6/nq3_dets" --workers 3 &
    done
    wait
    taskset -c 172-179 $PY -m jevdrive.nq3_feats tokens
  ) > "$R/feats/vjepa_yolo.log" 2>&1 & pids+=($!)
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  tail -n 3 "$R"/feats/*.log
  return $rc
}

feats_start() {   # feats in the background (own process group, PID in runs/nq3/c/pids.txt) so the Q2 pilot runs alongside
  mkdir -p "$R/feats"
  setsid bash -c "source '$REPO/scripts/nq3_c_steps.sh'; if feats; then touch '$R/feats/OK'; else touch '$R/feats/FAIL'; fi" \
    > "$R/feats/log.txt" 2>&1 < /dev/null &
  echo "$! feats (setsid, kill with kill -TERM -$!) $(date '+%F %T')" | tee -a "$R/pids.txt"
}

feats_wait() {
  while [[ ! -f $R/feats/OK && ! -f $R/feats/FAIL ]]; do sleep 60; done
  tail -n 5 "$R"/feats/*.log
  [[ -f $R/feats/OK ]]
}

q1_op() {   # openpilot Cinque / Lebowski native plan on all 605 P6 streams (cores 164-179)
  P5_SET=$P6 OMP_NUM_THREADS=2 retry 10 taskset -c 164-179 $OPPY scripts/p5_openpilot.py --arrays plan temporal \
    --out-sub op_streams_plan --workers 12
}


# ---------------------------------------------------------------- Q2 v0 pilot and the closed-loop head
Q2OUT=$R/q2_pilot/out
q2py() { OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 taskset -c 164-179 $PY -m jevdrive.nq3_q2 "$@"; }

q2py_m() { local m=$1; shift; OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 taskset -c 164-179 \
  $PY -c "import sys; from jevdrive import nq3_q2 as Q; Q.MODELS = ('$m',); sys.argv = ['nq3_q2'] + sys.argv[1:]; Q.main()" "$@"; }

q2_pilot() {   # LOCO (primary), A0-A3, seeds 0-2, Cinque first (READY is chosen on Cinque); Lebowski after READY
  retry 5 q2py_m cinque pilot --split loco --seeds 0,1,2 --arms A0,A1,A2,A3 --run "$Q2OUT" --tag _main
}
q2_pilot_leb() { retry 5 q2py_m lebowski pilot --split loco --seeds 0,1,2 --arms A0,A1,A2,A3 --run "$Q2OUT" --tag _leb; }

ready() {      # registered rule (nq3_q2.choose): LOCO, Cinque, arm passing rule 7 in all seeds with the highest mean
               # bypass flip, none -> A1; mode head A3 if it passes else A2. Refit on all v0, export, READY
  q2py export --run "$Q2OUT"
  cat "$DATA_DIR/runs/nq3/q2/closed_loop_head/READY"
  printf '{"ready": "%s"}' "$DATA_DIR/runs/nq3/q2/closed_loop_head/READY" > "$R/ready/outputs.json"
}

q2_controls() { q2py controls --split loco --seeds 0,1,2 --run "$Q2OUT"; }                         # backbone Delta streams
q2_route()    { retry 5 q2py pilot --split route --seeds 0,1,2 --arms A0,A1,A2,A3 --run "$Q2OUT" --tag _main; }
q2_a4()       { retry 5 q2py pilot --split loco --seeds 0 --arms A4 --run "$Q2OUT" --tag _A4; }     # vocabulary control
q2_report()   { q2py report --run "$Q2OUT"; }

# ---------------------------------------------------------------- Q1 readouts and judge
q1_heads() { OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 retry 5 taskset -c 164-179 $PY -m jevdrive.nq3_q1 heads; }

wait_for() {   # wait_for <file> <deadline YYYY-mm-dd HH:MM>: 0 when the file exists, 0 with a note at the deadline
  local f=$1 dl; dl=$(date -d "$2" +%s)
  while [[ ! -f $f ]] && (( $(date +%s) < dl )); do sleep 300; done
  [[ -f $f ]] && echo "found $f" || echo "deadline $2 passed without $f: continuing without it"
}

q1_wait_nav() { wait_for "$R/q1_nav/DONE" "2026-09-27 01:30"; }
q1_wait_alp() { wait_for "$R/q1_alp/DONE" "2026-09-27 08:30"; }
q1_judge()    { OMP_NUM_THREADS=16 taskset -c 164-179 $PY -m jevdrive.nq3_q1 judge; }

# ---------------------------------------------------------------- Q2 v1 (lane A's P6 v1)
V1=carla_p6_v1
wait_v1() {
  wait_for "$DATA_DIR/runs/nq3/a/v1/DONE" "2026-09-27 08:00"
  [[ -f $DATA_DIR/runs/nq3/a/v1/DONE ]] || touch "$R/wait_v1/NO_V1"
}

v1_prep() {    # exam frames, openpilot temporal streams, mode labels on v1
  [[ -f $R/wait_v1/NO_V1 ]] && { echo "no v1: skipped"; return 0; }
  set -e
  taskset -c 164-179 $PY -c "from jevdrive import nq3_p6 as J; J.exam_frames('$V1')"
  P5_SET=$V1 $PY -m jevdrive.p5_openpilot prepare
  P5_SET=$V1 OMP_NUM_THREADS=2 taskset -c 150-179 $OPPY scripts/p5_openpilot.py --arrays temporal --out-sub op_streams_vis --workers 16
  P5_SET=$V1 $PY -m jevdrive.p5_openpilot finalize --arrays temporal --sub op_streams_vis
  q2py labels --set $V1
}

q2_v1() {      # the same code on v1: LOCO, town hold-out (Town13), route; A5 only if v0's mirror flag fired
  [[ -f $R/wait_v1/NO_V1 ]] && { echo "no v1: skipped"; return 0; }
  set -e
  local arms=A0,A1,A2,A3
  if $PY - <<'PY'
import sys, pandas as pd
c = pd.read_csv("research/results/nq3/q2/criteria_loco.csv")
sys.exit(0 if c.filter(like="mirror").fillna("").astype(str).apply(lambda s: s.str.len() > 0).any().any() else 1)
PY
  then arms=$arms,A5,A5m; echo "v0 mirror flag fired: A5 / A5m added"; fi
  for sp in town loco route; do
    OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 taskset -c 150-179 $PY -m jevdrive.nq3_q2 pilot --set $V1 --split $sp \
      --seeds 0,1,2 --arms $arms --run "$R/q2_v1/out" --tag _main
  done
  OMP_NUM_THREADS=16 taskset -c 150-179 $PY -m jevdrive.nq3_q2 report --run "$R/q2_v1/out" \
    --results research/results/nq3/q2_v1
}
