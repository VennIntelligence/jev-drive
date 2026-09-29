#!/usr/bin/env bash
# NAVSIM raise arm N3 (todos/2026-09-30-navsim-raise.md): N2's configuration on 19 968 + 40 000 rows (+ alt slots if arm A
# passed its inclusion rule). Needs arm S's features and labels (raise/scale DONE_final) and arm A's altdev.json.
# scale-row FAM + FAM2 labels [-> alt plans + labels of the scale rows] -> dev grid -> refit -> navtest once -> navhard once.
#   scripts/tmux_run.sh rn3 env CUDA_VISIBLE_DEVICES=<card> CUDA_DEVICE_ORDER=PCI_BUS_ID scripts/navsim_raise_n3.sh <looks>
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd); cd "$repo"
LOOKS=${1:-6}; CPUS=${RAISE_CPUS:-155-183}; SP=${RAISE_SP:-8}; PROCS=${RAISE_PROCS:-12}
R=$DATA_DIR/runs/skill_pack/raise/n3; S=$DATA_DIR/runs/skill_pack/raise/scale; A=$DATA_DIR/runs/skill_pack/raise/alt
mkdir -p "$R"; rm -f "$R/ERROR"
E=$DATA_DIR/envs; OP=$E/openpilot/bin/python; JV=$E/jevdrive/bin/python; NAV=$E/navsim1/bin/python
st() { echo "$(date '+%F %T') $*" | tee -a "$R/STATUS"; }
die() { st "ERROR: $*"; echo "$*" > "$R/ERROR"; exit 1; }
[[ ${CUDA_VISIBLE_DEVICES:-} =~ ^[56]$ ]] || die "CUDA_VISIBLE_DEVICES must be 5 or 6"
[[ -f $S/DONE_final && -f $A/altdev.json ]] || die "needs arm S DONE_final and arm A altdev.json"
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
T() { taskset -c "$CPUS" nice -n 19 "$@"; }
scored() { ls "$DATA_DIR"/runs/navsim/eval/$1/*/*.csv > /dev/null 2>&1; }
CACHE=$DATA_DIR/runs/navsim/metric_cache/v1_navtrain; F=$DATA_DIR/runs/skill_pack/n1/feat
NEV=(env NAVSIM_DEVKIT_ROOT=$DATA_DIR/third_party/navsim-v1.1 NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NUPLAN_MAPS_ROOT=$DATA_DIR/datasets/navsim/maps
     OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore)
label() { T "${NEV[@]}" $NAV scripts/n1_score_native.py run "$CACHE" "$1" "$2" --procs 20 2>&1 | grep -av -i warn >> "$R/log.txt"; }

USE_ALT=$($JV -c "import json; r = json.load(open('$A/altdev.json')); print(int(r['alt=1']['pdms_hold'] - r['alt=0']['pdms_hold'] >= 0.2))")
st "alt inclusion rule: $USE_ALT"
if [[ ! -f $S/fam_score/done ]]; then
  st "scale-row FAM + FAM2 labels"
  T $JV -c "
import numpy as np; from jevdrive.navsim_raise import family, fam_list
z = np.load('$F/lb_n2train_n40000.npz'); np.savez('$S/cands_fam.npz', tokens=z['tokens'], cands=family(z['native'], fam_list(2)))" || die "fam cands"
  label "$S/cands_fam.npz" "$S/fam_score"; touch "$S/fam_score/done"
fi
if [[ $USE_ALT == 1 && ! -f $S/alt_score/done ]]; then
  [[ -f $F/lb_n2train__alt_n40000.npz ]] || { st "scale-row alt plans, $PROCS shards"
    T $OP scripts/n1_extract.py extract --alt --data lb_n2train --n 40000 --procs "$PROCS" >> "$R/log.txt" 2>&1 || die "alt extract"
    T $OP scripts/n1_extract.py merge --alt --data lb_n2train --n 40000 >> "$R/log.txt" 2>&1 || die "alt merge"; }
  T $JV -c "
from jevdrive.navsim_raise import alt_cands; import numpy as np
z = np.load('$F/lb_n2train__alt_n40000.npz'); np.savez('$S/cands_alt.npz', tokens=z['tokens'], cands=alt_cands('lb_n2train_n40000', out=''))" || die "alt cands"
  label "$S/cands_alt.npz" "$S/alt_score"; touch "$S/alt_score/done"
fi
[[ -f $R/n3dev.json ]] || { st "dev grid (held-out only)"; T $JV -m jevdrive.navsim_raise n3dev $([[ $USE_ALT == 1 ]] || echo --noalt) >> "$R/log.txt" 2>&1 || die n3dev; }
[[ -f $R/navhard_n3.npz ]] || { st "final refit"; T $JV -m jevdrive.navsim_raise n3final >> "$R/log.txt" 2>&1 || die n3final; }
scored v1_navtest_sp_n3_navtest || { st "score navtest N3 (the one test run)"
  NAVSIM_THREADS=$SP T scripts/navsim_zs_score.sh score v1 navtest sp_n3_navtest "$R/navtest_n3.npz" > "$R/score_navtest.log" 2>&1 || die "score navtest"; }
st "report"; T $JV -m jevdrive.navsim_raise report --arm n3 --looks "$LOOKS" > "$R/report.md" 2>> "$R/log.txt" || die report
scored v2_navhard_two_stage_sp_n3_navhard || { st "score navhard N3 (once)"
  NAVSIM_THREADS=$SP T scripts/navsim_zs_score.sh score v2 navhard_two_stage sp_n3_navhard "$R/navhard_n3.npz" > "$R/score_navhard.log" 2>&1 || die "score navhard"; }
st "DONE n3"; touch "$R/DONE"
