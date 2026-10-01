#!/usr/bin/env bash
# Full-benchmark run of the default op_interp pipeline (research/openpilot-openloop-integration.md section 9):
# Cinque native plan, base adapter (lever arm + linear resample, no retime -- user decision 2026-09-28), two
# interpolators (ego-motion warp on CPU, GIMM-VFI context-rate grid on GPU). Resumable: every step is skipped once
# its output exists. STATUS / DONE / ERROR in $DATA_DIR/runs/op_interp/<out>.
#
#   scripts/tmux_run.sh opi-navfull-pilot experiments/op_openloop/archive/op_interp_full.sh navtest 2000              # staging: same seed-0
#                                                                                               # 2000 tokens as the
#                                                                                               # existing nav/ subset
#   scripts/tmux_run.sh opi-navfull       experiments/op_openloop/archive/op_interp_full.sh navtest                    # full navtest, 12 146
#   scripts/tmux_run.sh opi-navhard       experiments/op_openloop/archive/op_interp_full.sh navhard_two_stage           # full navhard, 5 912
#
# split: navtest (-> out navfull, v1 PDMS) or navhard_two_stage (-> out navhard, v2 EPDMS -- the v1.1 devkit has no
# navhard_two_stage split). n (optional, default 0 = every token): caps the token count for a staging pilot; with
# split=navtest and the default seed 0, this reproduces the exact 2000-token subset the doc's calibration numbers
# (GIMM 84.7, warp 83.1) came from, as a regression check on the refactored nav-cache / nav-export / nav-report.
set -uo pipefail
repo=$(cd "$(dirname "$0")/../../.." && pwd); cd "$repo"
split=${1:?usage: op_interp_full.sh navtest|navhard_two_stage [n]}
n=${2:-0}
if [[ $split == navtest ]]; then out=${OUT:-navfull}; ver=v1; else out=${OUT:-navhard}; ver=v2; fi
[[ -n ${3:-} ]] && out=$3   # optional explicit --out override (3rd positional), e.g. for the pilot dir
R=$DATA_DIR/runs/op_interp/$out; mkdir -p "$R"
CPUS=${CPUS:-96-117,120-132,134-138}       # sched row op-interp-warp: warp synth + openpilot inference (CPU launch-bound)
CPUS_GIMM=${CPUS_GIMM:-170-171,174-176}    # sched row op-interp-gimm
GPU=${GPU:-2}                              # sched row op-interp-gimm: GPU2, <= 15 GB, after op-adapt navtrain cache (done 2026-09-28 18:05)
PROCS=${PROCS:-8}
export CUDA_VISIBLE_DEVICES=$GPU
OP="taskset -c $CPUS $DATA_DIR/envs/openpilot/bin/python experiments/op_openloop/lib/op_interp.py"
VF="taskset -c $CPUS_GIMM $DATA_DIR/envs/vfi/bin/python experiments/op_openloop/lib/op_interp.py"
PJ="taskset -c $CPUS $DATA_DIR/envs/jevdrive/bin/python experiments/op_openloop/lib/op_interp.py"
st() { echo "$(date '+%F %T') $*" | tee -a "$R/STATUS"; }
die() { st "ERROR $*"; echo "$*" > "$R/ERROR"; exit 1; }

st "start split=$split out=$out n=$n ver=$ver"
[[ -f $R/meta.json ]] || { st "nav-cache -> $out"; $OP nav-cache --split "$split" --out "$out" --n "$n" --workers 24 || die nav-cache; }
[[ -f $R/warp.npy ]] || { st "synth $out warp (CPU)"; $OP synth --data "$out" --method warp --workers 24 || die "synth warp"; }
[[ -f $R/gimm_g0.2.npy ]] || { st "synth $out gimm grid (GPU $GPU)"; $VF synth --data "$out" --method gimm --grid 0.2 --batch 8 --workers 12 || die "synth gimm"; }

run() { local f=$1; [[ -f $R/plans/$f@cinque.npz ]] && return 0; st "run $out $f@cinque"; $OP run --data "$out" --frames "$f" --model cinque --procs "$PROCS" || die "run $f"; }
run warp
run gimm_g0.2

st "nav-export (base only)"; $PJ nav-export --data "$out" --adapters base || die nav-export
st "score ($ver $split)"; experiments/op_openloop/archive/op_interp_score.sh "$CPUS" "$out" "$ver" "$split" > "$R/score.log" 2>&1 || die score
st "report"; $PJ nav-report --data "$out" --ver "$ver" --split "$split" --refs "warp-cinque__base" > "$R/report.log" 2>&1 || die report
st "DONE"; touch "$R/DONE"
