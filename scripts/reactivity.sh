#!/usr/bin/env bash
# todos/2026-09-25-reactivity-program.md on the shared box. Launch each mode through scripts/slot_run.sh as slot
# reactivity-<mode>; the P5 v1 chain (v1-*) is armed by scripts/reactivity_v1_arm.sh.
# v0 modes: 6 pinned cores (196-201), nice 10, GPU from slot_run (CUDA_VISIBLE_DEVICES).
# v1 modes (deviation 7): one expert set per call (P5_SET=carla_p5v1_<expert>); cards from V1_GPUS, CPUs picked at
# launch from a per-mode list, skipping every CPU a live process is pinned to, so modes never overlap each other or
# anyone else's pinned job.
set -uo pipefail
cd ~/data/jev-drive
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
PIN=(taskset -c 196-201 nice -n 10)
OP=$DATA_DIR/envs/openpilot/bin/python
PY=.venv/bin/python
SUB=op_streams_vis
V1_GPUS=${V1_GPUS:-"1 2 4"}
QWEN_PROCS=${QWEN_PROCS:-1}          # extraction processes per card (see the profiling subsection in the todo)
QWEN_CORES=${QWEN_CORES:-6}          # per process: main + 4 loader workers (a clip item is ~0.4 core-s)

free_cpus() {  # free_cpus <cpu list> <n>: the first n CPUs of the list that no live process is pinned to
    python3 - "$1" "$2" <<'EOF'
import os, sys
def parse(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out
busy = set()
for d in os.listdir("/proc"):
    try:
        for line in open("/proc/%s/status" % d):
            if line.startswith("Cpus_allowed_list"):
                c = parse(line.split(":")[1].strip())
                if len(c) < 100:          # pinned; an unpinned process is allowed on every CPU of the box
                    busy |= set(c)
    except (OSError, ValueError):
        continue
print(",".join(str(c) for c in [c for c in parse(sys.argv[1]) if c not in busy][:int(sys.argv[2])]))
EOF
}
roomiest_gpu() {  # the card of V1_GPUS with the most free memory
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
        awk -F', ' -v want="$V1_GPUS" 'BEGIN { n = split(want, a, " "); for (i = 1; i <= n; i++) ok[a[i]] = 1 }
                                       ($1 in ok) { print $2, $1 }' | sort -rn | head -1 | cut -d' ' -f2
}
note() { echo "$(date '+%Y-%m-%d %H:%M') [REACTIVITY/V1] $*" | tee -a "$DATA_DIR/runs/zeroshot-exam/gpu-plan.md"; }
v1_set() {
    [[ ${1:-} == pdm || ${1:-} == ba ]] || { echo "expert: pdm | ba" >&2; exit 2; }
    export P5_SET=${V1_PREFIX:-carla_p5v1_}$1          # V1_PREFIX: a dry-run copy of the set
    [[ -f $DATA_DIR/processed/$P5_SET/index.parquet ]] || { echo "no index for $P5_SET" >&2; exit 1; }
}
# disjoint candidate CPU lists per mode (and per expert where two can run at once)
declare -A CPUS=([qwen]=${QWEN_CPUS:-156-207} [op]=${OP_CPUS:-52-77} [exam-pdm]=${EXAM_CPUS:-78-90} [exam-ba]=${EXAM_CPUS:-91-103}
                 [mc-pdm]=${MC_CPUS:-104-113} [mc-ba]=${MC_CPUS:-114-131})

case $1 in
    # D0: `temporal`, `vision`, `hidden` on the P5 streams (same streams and renderer as experiment 1)
    d0-op)   "${PIN[@]}" $OP scripts/p5_openpilot.py --shard 0/2 --workers 5 --arrays temporal vision hidden --out-sub $SUB & a=$!
             "${PIN[@]}" $OP scripts/p5_openpilot.py --shard 1/2 --workers 5 --arrays temporal vision hidden --out-sub $SUB & b=$!
             wait $a; ra=$?; wait $b; rb=$?; (( ra == 0 && rb == 0 )) || exit 1
             "${PIN[@]}" $PY -m jevdrive.p5_openpilot finalize --arrays temporal,vision,hidden --sub $SUB ;;
    d0-exam) "${PIN[@]}" $PY -m jevdrive.p5_exam run --op cinque,lebowski --op-arrays temporal,vision,hidden --op-sub $SUB \
                 --heads-skip "op-cinque hidden" ;;   # deviation 6: a 16 384-d ridge is ~6 h of CPU eigh
    # M-C: dual-stream reaction head, pair / hard / uniform / single-stream arms (P5 v0 smoke)
    mc)      "${PIN[@]}" $PY -m jevdrive.reactivity_mc ;;
    mc-wide) "${PIN[@]}" $PY -m jevdrive.reactivity_mc --lams=-5,7 ;;   # post-hoc sensitivity

    # ---- P5 v1 (deviation 7): v1-<step> <pdm|ba>
    # Qwen L18: plan (reuse P4 and identical v0 clips), then QWEN_PROCS claiming processes per card, one retry pass
    v1-qwen) v1_set "${2:-}"
             taskset -c "$(free_cpus "${CPUS[qwen]}" 4)" $PY -m jevdrive.p5_qwen plan || exit 1
             for pass in 1 2; do
                 pids=()
                 for g in $V1_GPUS; do
                     for ((p = 0; p < QWEN_PROCS; p++)); do
                         c=$(free_cpus "${CPUS[qwen]}" "$QWEN_CORES")
                         [[ -z $c ]] && { echo "no free CPU in ${CPUS[qwen]}" >&2; continue; }
                         note "v1-qwen $2 pass $pass: worker on GPU $g, CPUs $c"
                         CUDA_VISIBLE_DEVICES=$g taskset -c "$c" nice -n 5 $PY -m jevdrive.p5_qwen work --batch 2 --workers 4 &
                         pids+=($!)
                         sleep 3          # its pin is visible in /proc before the next free_cpus
                     done
                 done
                 for p in "${pids[@]}"; do wait "$p"; done
                 $PY -m jevdrive.p5_qwen check && exit 0
             done
             exit 1 ;;
    # openpilot temporal / vision / hidden: plan, link v0's identical streams, two shards on one card, finalize
    v1-op)   v1_set "${2:-}"
             c=$(free_cpus "${CPUS[op]}" "${OP_CORES:-16}"); [[ -n $c ]] || { echo "no free CPU" >&2; exit 1; }
             g=${OP_GPU:-$(roomiest_gpu)}
             note "v1-op $2: GPU $g, CPUs $c"
             taskset -c "$c" $PY -m jevdrive.p5_openpilot prepare || exit 1
             taskset -c "$c" $PY -m jevdrive.p5_openpilot reuse --sub $SUB || exit 1
             w=$(( $(tr ',' '\n' <<< "$c" | wc -l) / 2 - 1 )); (( w < 2 )) && w=2
             CUDA_VISIBLE_DEVICES=$g taskset -c "$c" nice -n 5 $OP scripts/p5_openpilot.py --shard 0/2 --workers $w \
                 --arrays temporal vision hidden --out-sub $SUB & a=$!
             CUDA_VISIBLE_DEVICES=$g taskset -c "$c" nice -n 5 $OP scripts/p5_openpilot.py --shard 1/2 --workers $w \
                 --arrays temporal vision hidden --out-sub $SUB & b=$!
             wait $a; ra=$?; wait $b; rb=$?; (( ra == 0 && rb == 0 )) || exit 1
             taskset -c "$c" $PY -m jevdrive.p5_openpilot finalize --arrays temporal,vision,hidden --sub $SUB ;;
    # D0 exam and M-C, exactly the v0 calls (deviation 7), on the expert's set
    v1-exam) v1_set "${2:-}"
             c=$(free_cpus "${CPUS[exam-$2]}" 12); g=$(roomiest_gpu); [[ -n $c ]] || { echo "no free CPU" >&2; exit 1; }
             note "v1-exam $2: GPU $g, CPUs $c"
             OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=$g taskset -c "$c" nice -n 5 \
                 $PY -m jevdrive.p5_exam run --op cinque,lebowski --op-arrays temporal,vision,hidden --op-sub $SUB \
                 --heads-skip "op-cinque hidden" ;;
    v1-mc)   v1_set "${2:-}"
             c=$(free_cpus "${CPUS[mc-$2]}" 8); g=$(roomiest_gpu); [[ -n $c ]] || { echo "no free CPU" >&2; exit 1; }
             note "v1-mc $2: GPU $g, CPUs $c"
             OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$g taskset -c "$c" nice -n 5 \
                 $PY -m jevdrive.reactivity_mc --op-sub $SUB ;;
    v1-final) note "chain done: D0 exam + M-C on carla_p5v1_pdm and carla_p5v1_ba" \
                   "(runs/p5_pairs/exam-d0-carla_p5v1_*, runs/reactivity/mc-carla_p5v1_*)" ;;
    *) echo "mode: d0-op | d0-exam | mc | mc-wide | v1-qwen|v1-op|v1-exam|v1-mc <pdm|ba> | v1-final" >&2; exit 2 ;;
esac
