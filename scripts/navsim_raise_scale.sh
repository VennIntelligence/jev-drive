#!/usr/bin/env bash
# NAVSIM raise, scale-up arm S (todos/2026-09-30-navsim-raise.md): M more navtrain tokens for the N1b scorer.
# token list -> keyframes -> [CPU] E6-style per-anchor labels || [GPU] GIMM frames -> Cinque features -> native labels
# -> held-out fit [-> final: navtest scored once]. One whole GPU (default 5), resumable, STATUS / DONE_* / ERROR in
# $DATA_DIR/runs/skill_pack/raise/scale.
#   scripts/tmux_run.sh rscale env CUDA_VISIBLE_DEVICES=5 CUDA_DEVICE_ORDER=PCI_BUS_ID scripts/navsim_raise_scale.sh <M> [pilot|fit|final]
# pilot = first 192 tokens through the whole chain (no fit); fit = held-out numbers only; final = fit + the one navtest scoring.
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd); cd "$repo"
M=$1 MODE=${2:-fit}
GPU=${CUDA_VISIBLE_DEVICES:-}; CPUS=${RAISE_CPUS:-155-183}; GW=${RAISE_GIMM:-5}; PROCS=${RAISE_PROCS:-6}; AP=${RAISE_AP:-20}; SP=${RAISE_SP:-8}
R=$DATA_DIR/runs/skill_pack/raise/scale; mkdir -p "$R"; rm -f "$R/ERROR"
E=$DATA_DIR/envs; OP=$E/openpilot/bin/python; VFI=$E/vfi/bin/python; JV=$E/jevdrive/bin/python; NAV=$E/navsim1/bin/python
st() { echo "$(date '+%F %T') $*" | tee -a "$R/STATUS"; }
die() { st "ERROR: $*"; echo "$*" > "$R/ERROR"; exit 1; }
[[ $GPU =~ ^[0-9]$ ]] || die "set CUDA_VISIBLE_DEVICES to the one assigned card (got '$GPU')"
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
T() { taskset -c "$CPUS" nice -n 19 "$@"; }
F=$DATA_DIR/runs/skill_pack/n1/feat; D=lb_n2train; CACHE=$DATA_DIR/runs/navsim/metric_cache/v1_navtrain
NEV=(env NAVSIM_DEVKIT_ROOT=$DATA_DIR/third_party/navsim-v1.1 NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NUPLAN_MAPS_ROOT=$DATA_DIR/datasets/navsim/maps
     OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore)
[[ $MODE == pilot ]] && { N=192; SFX=_pilot; } || { N=$M; SFX=; }
(( N % 64 == 0 )) || die "N must be a multiple of 64"

[[ -f $R/tokens.txt ]] || { st "token list M=$M"; T $JV -m jevdrive.navsim_raise tokens2 --m "$M" >> "$R/log.txt" 2>&1 || die tokens2; }
head -n "$N" "$R/tokens.txt" > "$R/tokens_n$N.txt"
[[ -f $DATA_DIR/runs/op_lb/$D/meta.json ]] || { st "keyframes ($D, all M)"; T $OP scripts/n1_extract.py prep --data $D --tokens "$R/tokens.txt" --workers 16 >> "$R/log.txt" 2>&1 || die prep; }

st "anchor labels (CPU, $AP procs, background)"
T "${NEV[@]}" $NAV scripts/elicit_e6_score.py run "$CACHE" "$DATA_DIR/runs/elicitation/e6-prep/20260926-003758/anchors.npz" \
    "$R/tokens_n$N.txt" "$R/anchor_score$SFX" --procs "$AP" > "$R/anchor_score.log" 2>&1 &
apid=$!

st "GIMM frames, first $((N / 32)) chunks, $GW workers on GPU $GPU"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" -lms 10000 > "$R/vram$SFX.txt" &
vpid=$!
pids=()
for w in $(seq 1 "$GW"); do
  T $VFI scripts/n1_extract.py synth --data $D --gpu "$GPU" --limit-chunks $((N / 32)) >> "$R/synth_w$w.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || die "GIMM worker $p failed"; done
st "GIMM done"; kill "$vpid" 2>/dev/null
[[ -f $F/${D}_n$N.npz ]] || { st "Cinque features n=$N, $PROCS shards"
  T $OP scripts/n1_extract.py extract --data $D --n "$N" --procs "$PROCS" >> "$R/log.txt" 2>&1 || die extract
  T $OP scripts/n1_extract.py merge --data $D --n "$N" >> "$R/log.txt" 2>&1 || die merge; }
T $JV -c "
import numpy as np; from jevdrive.skill_pack_n0 import stretch; from jevdrive.skill_pack_n1 import SCALES
z = np.load('$F/${D}_n$N.npz'); c = np.stack([[stretch(p, s) for s in SCALES] for p in z['native']]).astype(np.float32)
np.savez('$R/cands_n$N.npz', tokens=z['tokens'], cands=c)" || die cands
st "native labels ($SP procs)"
T "${NEV[@]}" $NAV scripts/n1_score_native.py run "$CACHE" "$R/cands_n$N.npz" "$R/native_score$SFX" --procs "$SP" 2>&1 | grep -av -i warn >> "$R/log.txt"
wait "$apid" || die "anchor labels failed"
st "labels done"
if [[ $MODE == pilot ]]; then
  T $JV -m jevdrive.navsim_raise pilotcheck >> "$R/log.txt" 2>&1 || die "pilot checklist failed (pilot_check.json)"
  st "DONE pilot (checklist passed)"; touch "$R/DONE_pilot"; exit 0
fi

st "fit (held-out)"; [[ $MODE == final ]] && FL=--final || FL=
T $JV -m jevdrive.navsim_raise fit --arm scale $FL >> "$R/log.txt" 2>&1 || die fit
if [[ $MODE == final ]] && ! ls "$DATA_DIR"/runs/navsim/eval/v1_navtest_sp_scale_navtest/*/*.csv > /dev/null 2>&1; then
  st "score navtest (the one test run)"
  NAVSIM_THREADS=$SP T scripts/navsim_zs_score.sh score v1 navtest sp_scale_navtest "$DATA_DIR/runs/skill_pack/raise/scale/navtest_scale.npz" \
      > "$R/score_navtest.log" 2>&1 || die "score navtest"
fi
st "DONE $MODE n=$N"; touch "$R/DONE_$MODE"
