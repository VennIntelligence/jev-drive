#!/usr/bin/env bash
# NAVSIM raise, openpilot-alternatives arm (todos/2026-09-30-navsim-raise.md): Cinque rolled out on the GIMM frames with a
# forced desire (laneChange L / R, turn L / R, held from -1.0 s) -> 4 alternative native plans per token (train, navtest,
# navhard) -> x {1.00, 1.15} stretch -> devkit sub-score labels on the train tokens. GPU 6 (TensorRT, small).
#   scripts/tmux_run.sh ralt env CUDA_VISIBLE_DEVICES=6 CUDA_DEVICE_ORDER=PCI_BUS_ID scripts/navsim_raise_alt.sh
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd); cd "$repo"
CPUS=${RAISE_CPUS:-155-183}; PROCS=${RAISE_PROCS:-6}
R=$DATA_DIR/runs/skill_pack/raise/alt; mkdir -p "$R"; rm -f "$R/ERROR"
E=$DATA_DIR/envs; OP=$E/openpilot/bin/python; JV=$E/jevdrive/bin/python; NAV=$E/navsim1/bin/python
st() { echo "$(date '+%F %T') $*" | tee -a "$R/STATUS"; }
die() { st "ERROR: $*"; echo "$*" > "$R/ERROR"; exit 1; }
[[ ${CUDA_VISIBLE_DEVICES:-} == 6 ]] || die "CUDA_VISIBLE_DEVICES must be 6"
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
T() { taskset -c "$CPUS" nice -n 19 "$@"; }
F=$DATA_DIR/runs/skill_pack/n1/feat
for dn in lb_n1train:19968 lb_navtest:12146 lb_navhard:5912; do
  d=${dn%:*} n=${dn#*:}
  [[ -f $F/${d}__alt_n$n.npz ]] && continue
  st "alt plans $d n=$n, $PROCS shards"
  T $OP scripts/n1_extract.py extract --alt --data $d --n $n --procs "$PROCS" >> "$R/log.txt" 2>&1 || die "extract $d"
  T $OP scripts/n1_extract.py merge --alt --data $d --n $n >> "$R/log.txt" 2>&1 || die "merge $d"
done
T $JV -m jevdrive.navsim_raise alt_cands >> "$R/log.txt" 2>&1 || die cands
NEV=(env NAVSIM_DEVKIT_ROOT=$DATA_DIR/third_party/navsim-v1.1 NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NUPLAN_MAPS_ROOT=$DATA_DIR/datasets/navsim/maps
     OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore)
st "labels (devkit, v1_e6sub)"
T "${NEV[@]}" $NAV scripts/n1_score_native.py run "$DATA_DIR/runs/navsim/metric_cache/v1_e6sub" "$R/cands_train.npz" "$R/score_train" \
    --procs 14 2>&1 | grep -av -i warn >> "$R/log.txt"
(( $(ls "$R"/score_train/chunk_*.npz | wc -l) >= 400 )) || die "labels incomplete"
st "DONE alt"; touch "$R/DONE"
