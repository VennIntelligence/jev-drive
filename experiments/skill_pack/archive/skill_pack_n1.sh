#!/usr/bin/env bash
# Skill pack N1 chain (fc65452:todos/2026-09-29-n1-scorer.md): GIMM frames -> Cinque `temporal` features -> native-candidate labels
# -> scorer fit on held-out logs [-> the one navtest scoring]. GPU 6 only, resumable, writes STATUS / DONE_n<N> / ERROR.
#   scripts/tmux_run.sh n1 env CUDA_VISIBLE_DEVICES=6 CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4 experiments/skill_pack/archive/skill_pack_n1.sh <N> [test|final]
# N = number of E6 tokens (multiple of 64) from lb_n1train's fixed order; `test` also extracts the navtest features;
# `final` = `test` + fit --final + scores navtest exactly once + report.
set -uo pipefail
repo=$(cd "$(dirname "$0")/../../.." && pwd); cd "$repo"
N=$1 MODE=${2:-}
CPUS=${N1_CPUS:-155-167}; GW=${N1_GIMM:-2}; PROCS=${N1_PROCS:-6}; SP=${N1_SP:-8}
R=$DATA_DIR/runs/skill_pack/n1; mkdir -p "$R"; rm -f "$R/ERROR"
E=$DATA_DIR/envs; OP=$E/openpilot/bin/python; VFI=$E/vfi/bin/python; JV=$E/jevdrive/bin/python
st() { echo "$(date '+%F %T') $*" | tee -a "$R/STATUS"; }
die() { st "ERROR: $*"; echo "$*" > "$R/ERROR"; exit 1; }
[[ ${CUDA_VISIBLE_DEVICES:-} == 6 ]] || die "CUDA_VISIBLE_DEVICES must be 6 (got '${CUDA_VISIBLE_DEVICES:-}')"
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
T() { taskset -c "$CPUS" nice -n 19 "$@"; }
F=$DATA_DIR/runs/skill_pack/n1/feat
(( N % 64 == 0 )) || die "N must be a multiple of 64"

[[ -f $DATA_DIR/runs/op_lb/lb_n1train/meta.json ]] || { st "prep lb_n1train (keys)"; T $OP experiments/skill_pack/archive/n1_extract.py prep --workers 8 >> "$R/log.txt" 2>&1 || die prep; }

st "GIMM frames, first $((N / 32)) chunks, $GW workers on GPU 6"
pids=()
for w in $(seq 1 "$GW"); do
  T $VFI experiments/skill_pack/archive/n1_extract.py synth --gpu 6 --limit-chunks $((N / 32)) >> "$R/synth_w$w.log" 2>&1 &
  pids+=($!)
done
if [[ $MODE == test || $MODE == final ]] && [[ ! -f $F/lb_navtest_n12146.npz ]]; then
  st "navtest features (Cinque on the lane's GIMM cache), $PROCS shards"
  T $OP experiments/skill_pack/archive/n1_extract.py extract --data lb_navtest --n 12146 --procs "$PROCS" >> "$R/log.txt" 2>&1 || die "extract navtest"
  T $OP experiments/skill_pack/archive/n1_extract.py merge --data lb_navtest --n 12146 >> "$R/log.txt" 2>&1 || die "merge navtest"
fi
for p in "${pids[@]}"; do wait "$p" || die "GIMM worker $p failed"; done
st "GIMM done"

if [[ ! -f $F/lb_n1train_n$N.npz ]]; then
  st "train features n=$N, $PROCS shards"
  T $OP experiments/skill_pack/archive/n1_extract.py extract --data lb_n1train --n "$N" --procs "$PROCS" >> "$R/log.txt" 2>&1 || die "extract train"
  T $OP experiments/skill_pack/archive/n1_extract.py merge --data lb_n1train --n "$N" >> "$R/log.txt" 2>&1 || die "merge train"
fi
T $JV -m experiments.skill_pack.archive.skill_pack_n1 cands --n "$N" >> "$R/log.txt" 2>&1 || die cands

NAV=$E/navsim1/bin/python
NEV=(env NAVSIM_DEVKIT_ROOT=$DATA_DIR/third_party/navsim-v1.1 NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NUPLAN_MAPS_ROOT=$DATA_DIR/datasets/navsim/maps
     OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore)
if [[ ! -f $R/native_score_n$N/done ]]; then
  st "native-candidate labels (devkit, v1_e6sub cache), $SP procs"
  [[ -f $R/check.ok ]] || { T "${NEV[@]}" $NAV experiments/skill_pack/archive/n1_score_native.py check "$DATA_DIR/runs/navsim/metric_cache/v1_e6sub" \
      "$R/cands_train_n$N.npz" 2>&1 | grep -av -i warn | tee -a "$R/log.txt" | tail -2 && touch "$R/check.ok"; } || die "native check"
  T "${NEV[@]}" $NAV experiments/skill_pack/archive/n1_score_native.py run "$DATA_DIR/runs/navsim/metric_cache/v1_e6sub" "$R/cands_train_n$N.npz" \
      "$R/native_score_n$N" --procs "$SP" 2>&1 | grep -av -i warn >> "$R/log.txt"; [[ -f $R/native_score_n$N/chunk_00000.npz ]] || die "native score"
  touch "$R/native_score_n$N/done"
fi

st "fit (GPU 6, small)"
[[ $MODE == final ]] && FL=--final || FL=
T $JV -m experiments.skill_pack.archive.skill_pack_n1 fit --n "$N" $FL >> "$R/log.txt" 2>&1 || die fit
if [[ $MODE == final ]]; then
  if ! ls "$DATA_DIR"/runs/navsim/eval/v1_navtest_sp_n1_navtest/*/*.csv > /dev/null 2>&1; then
    st "score navtest (the one test run)"
    NAVSIM_THREADS=$SP T experiments/zeroshot_openloop/archive/navsim_zs_score.sh score v1 navtest sp_n1_navtest "$R/navtest_n1.npz" > "$R/score_navtest.log" 2>&1 || die "score navtest"
  fi
  st "report"; T $JV -m experiments.skill_pack.archive.skill_pack_n1 report > "$R/report.md" 2>> "$R/log.txt" || die report
fi
st "DONE n=$N $MODE"; touch "$R/DONE_n$N"
