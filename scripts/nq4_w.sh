#!/usr/bin/env bash
# Night queue 4, lane W (todos/2026-09-26-night-queue-4.md, W and its [E] entries): the latent world-model chain.
#   vjepa -> prep -> run seed 0 / 1 / 2 (5 folds each: train, probes, paired + action readouts) -> report
# Idempotent: a step with its .done marker is skipped, so re-running the script resumes. Each step is retried once;
# a second failure writes ERROR and stops. STATUS.md is rewritten at every step; DONE ends the chain.
# GPU: picked before every step - GPU 6 if it has >= 18 GB free, else an idle card among 5 / 4 / 3 (< 5 GB used, no
# CARLA process); it waits (60 s polls) until one is free, so a CARLA server that takes a borrowed card back is left alone.
# Cores 200-207 (taskset), 8 threads.   Start: scripts/tmux_run.sh nq4-w scripts/nq4_w.sh   (W_UNTIL=<step> stops after it)
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
D=$DATA_DIR/runs/nq4/w/chain
mkdir -p "$D"
exec 9>"$D/lock"
flock -n 9 || { echo "nq4-w: another chain holds $D/lock"; exit 1; }
[[ -f $D/DONE ]] && { echo "nq4-w: DONE already: $(cat "$D/DONE")"; exit 0; }
rm -f "$D/ERROR"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
CPUS=200-207
STEPS=(vjepa prep seed0 seed1 seed2 report)
declare -A EST=([vjepa]=30 [prep]=15 [seed0]=40 [seed1]=40 [seed2]=40 [report]=10)   # minutes; 2x = stop

status() {
    local s
    { echo "# nq4 W chain ($(date '+%F %H:%M:%S'))"; echo
      for s in "${STEPS[@]}"; do
          if [[ -f $D/$s.done ]]; then echo "- $s: done $(cat "$D/$s.done")"
          elif [[ $s == "${1:-}" ]]; then echo "- $s: RUNNING since $(date +%H:%M) on GPU ${2:-?} (estimate ${EST[$s]} min, stop at 2x)"
          else echo "- $s: pending (estimate ${EST[$s]} min)"; fi
      done; echo; echo "pid $$, log dirs under $DATA_DIR/runs/nq4/w/"; } > "$D/STATUS.md"
}

pick_gpu() {
    while true; do
        local used
        used=$(nvidia-smi -i 6 --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr -d ' ')
        if (( ${used#*,} - ${used%,*} >= 18000 )); then echo 6; return; fi
        for g in 5 4 3; do
            local uuid u
            uuid=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
            u=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
            if (( u < 5000 )) && ! nvidia-smi --query-compute-apps=gpu_uuid,name --format=csv,noheader | grep "$uuid" | grep -q CarlaUE4; then
                echo $g; return
            fi
        done
        sleep 60
    done
}

cmd_of() {
    case $1 in
        vjepa) echo "-m jevdrive.nq4_w vjepa --workers 7 --batch 64" ;;
        prep) echo "-m jevdrive.nq4_w prep --workers 8" ;;
        seed*) echo "-m jevdrive.nq4_w run --seed ${1#seed}" ;;
        report) echo "-m jevdrive.nq4_w report" ;;
    esac
}

for s in "${STEPS[@]}"; do
    [[ -f $D/$s.done ]] && continue
    ok=0
    for attempt in 1 2; do
        gpu=$(pick_gpu)
        status "$s" "$gpu"
        echo "$(date '+%F %H:%M:%S') step $s attempt $attempt on GPU $gpu" | tee -a "$D/chain.log"
        t0=$(date +%s)
        # shellcheck disable=SC2046
        CUDA_VISIBLE_DEVICES=$gpu timeout $(( EST[$s] * 2 * 60 )) taskset -c $CPUS .venv/bin/python $(cmd_of "$s") \
            > >(tee -a "$D/$s.out") 2>&1
        rc=$?
        el=$(( ($(date +%s) - t0) / 60 ))
        if (( rc == 0 )); then
            echo "$(date '+%H:%M') after ${el} min on GPU $gpu" > "$D/$s.done"; ok=1; break
        fi
        echo "$(date '+%F %H:%M:%S') step $s attempt $attempt failed rc=$rc after ${el} min (124 = over 2x the estimate)" | tee -a "$D/chain.log"
    done
    if (( ! ok )); then
        echo "step $s failed twice (last rc=$rc); see $D/$s.out and $D/chain.log" > "$D/ERROR"
        status; exit 1
    fi
    if [[ ${W_UNTIL:-} == "$s" ]]; then status; echo "nq4-w: stopping after $s (W_UNTIL)"; exit 0; fi
done
status
echo "$(date '+%F %H:%M:%S') report in research/results/nq4/w/ (box copy of the repo)" > "$D/DONE"
echo "nq4-w: DONE"
