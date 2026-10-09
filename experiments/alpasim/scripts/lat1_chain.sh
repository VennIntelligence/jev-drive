#!/usr/bin/env bash
# LAT1 (GPU slot warp, results/lat1_frame_synthesis.md): every box job of the lane as pool jobs, chained. Run on the box:
#   [STAGES="msgs equiv load loop"] [SYNC=1] bash experiments/alpasim/scripts/lat1_chain.sh [tag]     (default tag P2H10-F-s0)
# Output: $DATA_DIR/runs/alpasim/lat1. STAGES picks the stages to submit (default all four).
# msgs   driver-side messages of the 400-scene SH30 run (c0/sh30_400_rerun), CPU only
# equiv  lat1_check.py remap + prof + equiv on all of them
# load   lat1_check.py load on the first 64 scenes, 8 and 2 concurrent streams, one variant after the other on the same cores: cpu, gpu,
#        gpu without stage syncs, gpu with pack_gpu
# loop   closed loop on m1/lists/s1.txt as M1's s1-p2h10 run (same list, one chunk, same overrides): SH30_SYNTH=gpu (SH30_STAGE_SYNC=$SYNC),
#        then cpu
# Each stage writes <stage>.DONE; a failed pool job leaves its log in pool_<stage>/log.txt.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); repo=$(cd "$here/../../.." && pwd); cd "$repo"
tag=${1:-P2H10-F-s0}; R=$DATA_DIR/runs/alpasim; L=$R/lat1; PY=$DATA_DIR/envs/op-train/bin/python; C=experiments/alpasim/scripts/lat1_check.py
mkdir -p "$L"
env="env ALPASIM_SRC=$DATA_DIR/third_party/alpasim OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 SH30_TAG=$tag"
sub() { name=$1; shift; rm -rf "$L/pool_$name" "$L/$name.DONE"; .venv/bin/python -m jevdrive.cl submit --name "lat1-$name" --log-dir "$L/pool_$name" "$@" | tail -1; }
want() { [[ " ${STAGES:-msgs equiv load loop} " == *" $1 "* ]]; }
m=""
if want msgs; then
  run=$(ls -d "$R"/c0/sh30_400_rerun/*/ | tail -1); ls "${run}rollouts" > "$L/scenes400.txt"
  m=$(sub msgs --vram 0.5 --cpu 6 --ram 16 -- bash -c "rm -rf $L/msgs400.tmp; $DATA_DIR/third_party/alpasim/.venv/bin/python $here/c1_extract.py msgs \
--run ${run%/} --scenes $L/scenes400.txt --out $L/msgs400.tmp --jobs 6 && rm -rf $L/msgs400 && mv $L/msgs400.tmp $L/msgs400 && touch $L/msgs.DONE")
fi
if want equiv; then
  sub equiv ${m:+--after "$m"} --vram 9 --cpu 4 --ram 16 -- bash -c "$env $PY $C remap --msgs $L/msgs400 --out $L/remap.json --n 100 && \
$env $PY $C prof --msgs $L/msgs400 --out $L/prof.json && $env $PY $C equiv --msgs $L/msgs400 --out $L/equiv.json && touch $L/equiv.DONE"
fi
if want load; then
  ld=""; for v in "cpu_s8 --synth cpu" "gpu_s8 --synth gpu" "gpu_nosync_s8 --synth gpu --nosync" "gpu_packgpu_s8 --synth gpu --pack gpu" \
    "cpu_s2 --synth cpu --streams 2" "gpu_s2 --synth gpu --streams 2" "gpu_nosync_s2 --synth gpu --nosync --streams 2"; do
    ld+="$env $PY $C load --msgs $L/msgs400 --out $L/load_${v%% *}.json ${v#* } --n 64 && "; done
  sub load ${m:+--after "$m"} --vram 6 --cpu 8 --ram 16 -- bash -c "$ld $PY $C same --msgs $L/load_cpu_s8.json --out $L/load_gpu_s8.json > $L/load_same.json && \
$PY $C same --msgs $L/load_cpu_s8.json --out $L/load_gpu_nosync_s8.json >> $L/load_same.json && touch $L/load.DONE"
fi
if want loop; then
  ov="+e2e_challenge_nuplan=full runtime.nr_workers=2 runtime.endpoints.renderer.n_concurrent_rollouts=8 runtime.endpoints.driver.n_concurrent_rollouts=8 \
runtime.endpoints.controller.n_concurrent_rollouts=8 defines.nre_cache_size=9"
  prev=""; for k in gpu cpu; do
    D=$L/runs/s1-$k/$(date +%Y%m%d-%H%M%S); mkdir -p "$D"
    prev=$(sub "loop-$k" ${prev:+--after "$prev"} --vram 32 --cpu 8 --ram 25 --timeout-h 3 -- bash -c \
"env SH30_TAG=$tag SH30_SYNTH=$k SH30_STAGE_SYNC=${SYNC:-1} bash experiments/alpasim/scripts/run.sh $D sh30 --scene-list $R/m1/lists/s1.txt $ov && touch $L/loop-$k.DONE")
    echo "loop-$k $prev $D"
  done
fi
