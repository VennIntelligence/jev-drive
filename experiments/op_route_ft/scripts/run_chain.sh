#!/usr/bin/env bash
# op_route_ft chains (plans/2026-10-05-route-ft-prereg.md). One phase per call; STATUS / DONE / ERROR in $R/chain/<phase>/.
#   run_chain.sh pilot <gpu> <cpus>                 wait for the re-rendered CARLA pack, bank it, pilot rc-bear (400 steps), open-loop readouts
#   run_chain.sh arm <arm> <gpu> <cpus> [steps]     one full arm, then its open-loop readouts
#   run_chain.sh eval <gpu> <cpus> <models...>      open-loop readouts of finished models (and O)
set -uo pipefail
cd "$(dirname "$0")/../../.."
R=$DATA_DIR/runs/op_route_ft
PY=$DATA_DIR/envs/op-train/bin/python
phase=$1; shift
case $phase in
  pilot) name=pilot; gpu=$1; cpus=$2 ;;
  arm) name=arm-$1; arm=$1; gpu=$2; cpus=$3; steps=${4:-0} ;;
  eval) name=eval-$(date +%H%M); gpu=$1; cpus=$2; shift 2; models=("$@") ;;
  *) echo "unknown phase $phase"; exit 2 ;;
esac
D=$R/chain/$name; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
say() { echo "$(date '+%F %T') $name: $*" | tee -a "$D/log.txt" > /dev/null; echo "$(date '+%F %T') $name: $*" > "$D/STATUS"; }
die() { say "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
run() { CUDA_VISIBLE_DEVICES=$gpu taskset -c "$cpus" "$PY" experiments/op_route_ft/scripts/rft.py "$@" >> "$D/log.txt" 2>&1; }
PK=$DATA_DIR/runs/op_route_cmd/carla_pairs_s10000ol/packed
case $phase in
  pilot)
    say "waiting for $PK"
    until [[ -f $PK/route.npz && -f $PK/samples/route_carla/tab.npz && -f $PK/pack_summary.json ]]; do sleep 60; done
    say "bank ol"; run bank --which ol || die "bank"
    say "pilot rc-bear 400 steps"; run train --arm rc-bear --steps 400 --tag pilot-bear --fresh || die "pilot train"
    say "readouts"; run evalol --models O pilot-bear --carla ol || die "evalol" ;;
  arm)
    tag=$arm-s0
    say "train $arm"; if (( steps > 0 )); then run train --arm "$arm" --steps "$steps" || die "train"; else run train --arm "$arm" || die "train"; fi
    say "readouts"; run evalol --models "$tag" --carla ol || die "evalol" ;;
  eval)
    say "readouts ${models[*]}"; run evalol --models "${models[@]}" --carla ol || die "evalol" ;;
esac
say "done"; date '+%F %T' > "$D/DONE"
