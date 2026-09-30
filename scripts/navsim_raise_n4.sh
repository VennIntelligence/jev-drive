#!/usr/bin/env bash
# NAVSIM raise arm N4 (todos/2026-09-30-navsim-raise.md, "N4" preregistration): N3's configuration on every navtrain row
# outside the held-out logs (+24 064 rows on top of N3's 59 968). Resumable; STATUS / DONE_* / ERROR in $DATA_DIR/runs/skill_pack/raise/scale2.
# tokens -> keyframes -> [CPU] anchor labels || [GPU, one whole card] GIMM frames -> Cinque features -> native labels
# -> pilot checklist | (FAM + FAM2 labels -> held-out dev -> refit -> navtest once -> report -> navhard once).
#   scripts/tmux_run.sh rn4 env CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID scripts/navsim_raise_n4.sh pilot|final
# The GPU stage waits until the card has < 5 GB in use (or $R/GPU_FREE exists).
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd); cd "$repo"
MODE=${1:-pilot}; M=24064
GPU=${CUDA_VISIBLE_DEVICES:-}; CPUS=${RAISE_CPUS:-108-143}; GW=${RAISE_GIMM:-5}; PROCS=${RAISE_PROCS:-6}; AP=${RAISE_AP:-16}; SP=${RAISE_SP:-8}
R=$DATA_DIR/runs/skill_pack/raise/scale2; mkdir -p "$R"; rm -f "$R/ERROR"
E=$DATA_DIR/envs; OP=$E/openpilot/bin/python; VFI=$E/vfi/bin/python; JV=$E/jevdrive/bin/python; NAV=$E/navsim1/bin/python
st() { echo "$(date '+%F %T') $*" | tee -a "$R/STATUS"; }
die() { st "ERROR: $*"; echo "$*" > "$R/ERROR"; exit 1; }
[[ $GPU == 0 ]] || die "N4 runs on GPU 0 only (got '$GPU')"
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
T() { taskset -c "$CPUS" nice -n 19 "$@"; }
scored() { ls "$DATA_DIR"/runs/navsim/eval/$1/*/*.csv > /dev/null 2>&1; }
F=$DATA_DIR/runs/skill_pack/n1/feat; D=lb_n3train; CACHE=$DATA_DIR/runs/navsim/metric_cache/v1_navtrain
NEV=(env NAVSIM_DEVKIT_ROOT=$DATA_DIR/third_party/navsim-v1.1 NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NUPLAN_MAPS_ROOT=$DATA_DIR/datasets/navsim/maps
     OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore)
label() { T "${NEV[@]}" $NAV scripts/n1_score_native.py run "$CACHE" "$1" "$2" --procs "$SP" 2>&1 | grep -av -i warn >> "$R/log.txt"; }
[[ $MODE == pilot ]] && { N=192; SFX=_pilot; } || { N=$M; SFX=; }

[[ -f $R/tokens.txt ]] || { st "token list (skip arm S's 40000, take $M)"; T $JV -m jevdrive.navsim_raise tokens2 --m "$M" --skip 40000 --tag scale2 >> "$R/log.txt" 2>&1 || die tokens2; }
[[ $(wc -l < "$R/tokens.txt") == "$M" ]] || die "token list has $(wc -l < "$R/tokens.txt") rows, expected $M"
head -n "$N" "$R/tokens.txt" > "$R/tokens_n$N.txt"
[[ -f $DATA_DIR/runs/op_lb/$D/meta.json ]] || { st "keyframes ($D, all $M)"; T $OP scripts/n1_extract.py prep --data $D --tokens "$R/tokens.txt" --workers 16 >> "$R/log.txt" 2>&1 || die prep; }

st "anchor labels (CPU, $AP procs, background)"
T "${NEV[@]}" $NAV scripts/elicit_e6_score.py run "$CACHE" "$DATA_DIR/runs/elicitation/e6-prep/20260926-003758/anchors.npz" \
    "$R/tokens_n$N.txt" "$R/anchor_score$SFX" --procs "$AP" > "$R/anchor_score.log" 2>&1 &
apid=$!

if [[ ! -f $F/${D}_n$N.npz ]]; then
  while :; do   # whole-card gate: WL-2 CARLA servers are being drained from GPU 0
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
    { (( used < 5000 )) || [[ -f $R/GPU_FREE ]]; } && break
    sleep 120
  done
  st "GPU 0 free (${used} MiB used); GIMM frames, first $((N / 32)) chunks, $GW workers"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 -lms 10000 > "$R/vram$SFX.txt" &
  vpid=$!
  pids=()
  for w in $(seq 1 "$GW"); do
    T $VFI scripts/n1_extract.py synth --data $D --gpu 0 --limit-chunks $((N / 32)) >> "$R/synth_w$w.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || die "GIMM worker $p failed"; done
  st "GIMM done"; kill "$vpid" 2>/dev/null
  st "Cinque features n=$N, $PROCS shards"
  T $OP scripts/n1_extract.py extract --data $D --n "$N" --procs "$PROCS" >> "$R/log.txt" 2>&1 || die extract
  T $OP scripts/n1_extract.py merge --data $D --n "$N" >> "$R/log.txt" 2>&1 || die merge
fi
T $JV -c "
import numpy as np; from jevdrive.skill_pack_n0 import stretch; from jevdrive.skill_pack_n1 import SCALES
z = np.load('$F/${D}_n$N.npz'); c = np.stack([[stretch(p, s) for s in SCALES] for p in z['native']]).astype(np.float32)
np.savez('$R/cands_n$N.npz', tokens=z['tokens'], cands=c)" || die cands
st "native labels ($SP procs)"
label "$R/cands_n$N.npz" "$R/native_score$SFX"
wait "$apid" || die "anchor labels failed"
st "labels done"
if [[ $MODE == pilot ]]; then
  T $JV -m jevdrive.navsim_raise pilotcheck --tag scale2 >> "$R/log.txt" 2>&1 || die "pilot checklist failed (pilot_check.json)"
  st "DONE pilot (checklist passed)"; touch "$R/DONE_pilot"; exit 0
fi

if [[ ! -f $R/fam_score/done ]]; then
  st "FAM + FAM2 labels"
  T $JV -c "
import numpy as np; from jevdrive.navsim_raise import family, fam_list
z = np.load('$F/${D}_n$M.npz'); np.savez('$R/cands_fam.npz', tokens=z['tokens'], cands=family(z['native'], fam_list(2)))" || die "fam cands"
  label "$R/cands_fam.npz" "$R/fam_score"; touch "$R/fam_score/done"
fi
N4=$DATA_DIR/runs/skill_pack/raise/n4; mkdir -p "$N4"
[[ -f $N4/n4dev.json ]] || { st "N4 held-out (fixed N3 config)"; T $JV -m jevdrive.navsim_raise n4dev >> "$N4/log.txt" 2>&1 || die n4dev; }
[[ -f $N4/navhard_n4.npz ]] || { st "final refit"; T $JV -m jevdrive.navsim_raise n4final >> "$N4/log.txt" 2>&1 || die n4final; }
scored v1_navtest_sp_n4_navtest || { st "score navtest N4 (the one test run)"
  NAVSIM_THREADS=$SP T scripts/navsim_zs_score.sh score v1 navtest sp_n4_navtest "$N4/navtest_n4.npz" > "$N4/score_navtest.log" 2>&1 || die "score navtest"; }
st "report"; T $JV -m jevdrive.navsim_raise report --arm n4 --looks 7 > "$N4/report.md" 2>> "$N4/log.txt" || die report
scored v2_navhard_two_stage_sp_n4_navhard || { st "score navhard N4 (once, description)"
  NAVSIM_THREADS=$SP T scripts/navsim_zs_score.sh score v2 navhard_two_stage sp_n4_navhard "$N4/navhard_n4.npz" > "$N4/score_navhard.log" 2>&1 || die "score navhard"; }
st "DONE final"; touch "$R/DONE_final"
