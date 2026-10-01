#!/usr/bin/env bash
# Full NAVSIM v1.1 navtrain PDM metric cache (103 288 tokens, 1 192 logs), CPU only, devkit as shipped
# (same devkit, env and OPENBLAS workaround as experiments/zeroshot_openloop/archive/navsim_zs_score.sh, so it is interchangeable with v1_navtrain_oplb).
#   experiments/skill_pack/archive/navtrain_mcache.sh pilot1|pilot10|full
# Cache: $DATA_DIR/runs/navsim/metric_cache/v1_navtrain (existing files are skipped, never overwritten).
# Run dir: $DATA_DIR/runs/navtrain_mcache/<stage>-<ts>/ with log.txt, events.jsonl, DONE | ERROR.
# NMC_CPUS = taskset list, NMC_WORKERS = ray workers, NMC_NICE (default 19).
set -uo pipefail
stage=${1:?stage}
repo=$(cd "$(dirname "$0")/../../.." && pwd)
dk=$DATA_DIR/third_party/navsim-v1.1 py=$DATA_DIR/envs/navsim1/bin/python
cache=$DATA_DIR/runs/navsim/metric_cache/v1_navtrain
run=$DATA_DIR/runs/navtrain_mcache/$stage-$(date +%Y%m%d-%H%M%S); mkdir -p "$run"
cpus=${NMC_CPUS:?NMC_CPUS}; workers=${NMC_WORKERS:?NMC_WORKERS}
ev() { python3 -c "import json,sys,time; print(json.dumps({'t':time.time(),'kind':sys.argv[1],**json.loads(sys.argv[2])}))" "$1" "${2:-{\}}" >> "$run/events.jsonl"; }
exec > >(tee -a "$run/log.txt") 2>&1
fail() { echo "ERROR: $*"; ev error "{\"msg\": \"$*\"}"; echo "$*" > "$run/ERROR"; exit 1; }

export NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NUPLAN_MAPS_ROOT=$DATA_DIR/datasets/navsim/maps
export OPENSCENE_DATA_ROOT=$DATA_DIR/datasets/navsim NAVSIM_EXP_ROOT=$DATA_DIR/runs/navsim/eval
export NAVSIM_DEVKIT_ROOT=$dk OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$py" -c "import numpy as np; A=np.random.default_rng(0).normal(size=(40,40)); e=abs(A@np.linalg.inv(A)-np.eye(40)).max(); assert e<1e-8, f'BLAS broken: {e}'" || fail "BLAS check"

flt=()
if [[ $stage != full ]]; then
  n=${stage#pilot}
  logs=$("$py" - "$n" <<'PY'
import sys, yaml, os
f = os.path.expandvars("$DATA_DIR/third_party/navsim-v1.1/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml")
L = yaml.safe_load(open(f))["log_names"]; n = int(sys.argv[1])
print(",".join(f"'{L[i * len(L) // n + len(L) // (2 * n)]}'" for i in range(n)))
PY
  ) || fail "log pick"
  flt=("train_test_split.scene_filter.log_names=[$logs]" "train_test_split.scene_filter.tokens=null")
  [[ $n -lt $workers ]] && workers=$n
fi
ev start "{\"stage\": \"$stage\", \"cache\": \"$cache\", \"workers\": $workers, \"cpus\": \"$cpus\"}"
echo "stage $stage workers $workers cpus $cpus cache $cache run $run"; df -h "$DATA_DIR" | tail -1

# progress: cache file count every 120 s
( while sleep 120; do n=$(find "$cache" -name metric_cache.pkl 2>/dev/null | wc -l); ev progress "{\"files\": $n}"; echo "progress: $n pkl"; done ) &
prog=$!
t0=$(date +%s)
nice -n "${NMC_NICE:-19}" taskset -c "$cpus" "$py" "$dk/navsim/planning/script/run_metric_caching.py" train_test_split=navtrain \
  cache.cache_path=$cache worker=ray_distributed_no_torch worker.threads_per_node=$workers "${flt[@]}" \
  2>&1 | grep -av "Processing scenario\|Extracted .* scenarios for thread"
rc=${PIPESTATUS[0]}
kill "$prog" 2>/dev/null
dt=$(( $(date +%s) - t0 ))
n=$(find "$cache" -name metric_cache.pkl | wc -l)
ev end "{\"rc\": $rc, \"seconds\": $dt, \"files\": $n}"
echo "rc $rc, $dt s, $n pkl in cache"
(( rc == 0 )) || fail "run_metric_caching rc=$rc"
touch "$run/DONE"
