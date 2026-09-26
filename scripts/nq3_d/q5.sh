#!/usr/bin/env bash
# Night queue 3, Q5 (lane D): hack audit of DrivoR / WA-JEPA and the selectivity table, one unattended pass
# (todos/2026-09-26-night-queue-3.md, [D] 16:50 Q5 entry). Resumable: every finished artefact is skipped on a rerun.
#   requests + arms -> rig-perturbation images -> DrivoR (nusc, navtest, rig) -> WA-JEPA (fp32 navtest check, nusc,
#   navtest, rig) in the background while the DrivoR navtest arms are scored -> WA-JEPA arms scored -> equivalence
#   checks -> tables in research/results/nq3/q5/ (box repo copy).
# Resources: GPU $GPU (default 6; DrivoR ~1 GB, WA-JEPA <= ~12 GB at WJ_BS), every process pinned to $CPUS (180-199),
# BLAS threads 1 per devkit ray worker. Usage (in tmux): scripts/nq3_d/q5.sh
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
repo=$(cd "$(dirname "$0")/../.." && pwd)
GPU=${GPU:-6}; CPUS=${CPUS:-180-199}; WJ_BS=${WJ_BS:-8}; SCORE_THREADS=${SCORE_THREADS:-14}
R=$DATA_DIR/runs/nq3/q5; L=$R/logs; mkdir -p "$R/preds" "$R/frag" "$L"
pin=(taskset -c "$CPUS")
py=("${pin[@]}" env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$repo/.venv/bin/python" -m jevdrive.nq3_q5)
say() { echo "[$(date +%T)] q5: $*"; }
say "estimate (GPU 6 shared): DrivoR 11 arms x 16 782 requests ~0.5 h; WA-JEPA 12 arms x 16 782 at bs $WJ_BS ~${WJ_EST:-3} h;" \
    "devkit v1.1 23 navtest jobs ~1.5 h (overlaps WA-JEPA); rig perturbation 256 tokens x 6 arms ~20 min; total ~${Q5_EST:-4} h"

[[ -f $R/req/arms.json ]] || { "${py[@]}" req; "${py[@]}" arms; }
[[ -f $R/frag/arms_wajepa.npz ]] || "${py[@]}" frag-prep --workers 16

drivor() {   # set request arms out
  [[ -f $4 ]] && return 0
  say "DrivoR $1 -> $4"
  (cd "$DATA_DIR/third_party/drivor" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2 "${pin[@]}" "$DATA_DIR/envs/drivor/bin/python" \
    "$repo/scripts/nq3_d/drivor_arms.py" "$2" "$3" --out "$4" --workers "${5:-10}" > "$L/drivor_$1.log" 2>&1)
}
wajepa() {   # set request arms out [extra args]
  [[ -f $4 ]] && return 0
  say "WA-JEPA $1 -> $4"
  (cd "$DATA_DIR/third_party/wajepa" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 "${pin[@]}" "$DATA_DIR/envs/wajepa/bin/python" \
    "$repo/scripts/nq3_d/wajepa_arms.py" "$2" "$3" --out "$4" "${@:5}" > "$L/wajepa_$1.log" 2>&1)
}
score() {    # jobs file
  local ver split name npz
  while read -r ver split name npz; do
    [[ -z $ver ]] && continue
    if compgen -G "$DATA_DIR/runs/navsim/eval/${ver}_${split}_${name}/*/*.csv" > /dev/null; then continue; fi
    say "devkit $name"
    NAVSIM_THREADS=$SCORE_THREADS "${pin[@]}" "$repo/scripts/navsim_zs_score.sh" score "$ver" "$split" "$name" "$npz" \
      > "$L/score_$name.log" 2>&1 || { say "devkit FAILED $name (see $L/score_$name.log)"; return 1; }
  done < "$1"
}
frag() {     # model runner
  local a
  for a in orig ident yaw+0.5 yaw-0.5 z+5cm z-5cm; do
    "$2" "frag_$a" "$R/frag/req_$a.npz" "$R/frag/arms_$1.npz" "$R/frag/pred_$1_$a.npz" "${@:3}"
  done
}

drivor nusc "$R/req/nusc.npz" "$R/req/nusc_drivor_arms.npz" "$R/preds/nusc_drivor.npz"
drivor navtest "$R/req/navtest.npz" "$R/req/navtest_drivor_arms.npz" "$R/preds/navtest_drivor.npz"
frag drivor drivor
(
  wajepa navcheck "$R/req/navcheck.npz" "$R/req/navcheck_arms.npz" "$R/preds/navcheck_wajepa_fp32.npz" --no-amp --workers 2
  wajepa nusc "$R/req/nusc.npz" "$R/req/nusc_wajepa_arms.npz" "$R/preds/nusc_wajepa.npz" --bs "$WJ_BS" --workers 6
  wajepa navtest "$R/req/navtest.npz" "$R/req/navtest_wajepa_arms.npz" "$R/preds/navtest_wajepa.npz" --bs "$WJ_BS" --workers 6
  frag wajepa wajepa --bs "$WJ_BS" --workers 4
) > "$L/wajepa_chain.log" 2>&1 &
wj=$!
"${py[@]}" nav-jobs --model drivor
score "$R/nav/jobs_drivor.txt" || { kill "$wj" 2>/dev/null; exit 1; }
say "DrivoR scored; waiting for WA-JEPA (pid $wj)"
wait "$wj" || { say "WA-JEPA chain FAILED (see $L/wajepa_chain.log)"; exit 1; }
"${py[@]}" nav-jobs --model wajepa
score "$R/nav/jobs_wajepa.txt"
"${py[@]}" verify
"${py[@]}" tables
touch "$R/DONE"
say "done -> $repo/research/results/nq3/q5"
