#!/usr/bin/env bash
# LAT1 (GPU slot warp, results/lat1_frame_synthesis.md): every box job of the lane as pool jobs, chained. Run once on the box:
#   bash experiments/alpasim/scripts/lat1_chain.sh [tag]            (default P2H10-F-s0; output $DATA_DIR/runs/alpasim/lat1)
# msgs   driver-side messages of the 400-scene SH30 run (c0/sh30_400_rerun), CPU only
# equiv  lat1_check.py remap + prof + equiv on all of them
# load   lat1_check.py load, cpu and gpu, 8 and 2 concurrent streams on the first 64 scenes, one after the other on the same cores
# loop   closed loop on m1/lists/s1.txt as M1's s1-p2h10 run (same list, one chunk, same overrides): SH30_SYNTH=gpu, then cpu
# Each stage writes <stage>.DONE; a failed pool job leaves its log in pool_<stage>/log.txt.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); repo=$(cd "$here/../../.." && pwd); cd "$repo"
tag=${1:-P2H10-F-s0}; R=$DATA_DIR/runs/alpasim; L=$R/lat1; PY=$DATA_DIR/envs/op-train/bin/python; C=experiments/alpasim/scripts/lat1_check.py
mkdir -p "$L"
env="env ALPASIM_SRC=$DATA_DIR/third_party/alpasim OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 SH30_TAG=$tag"
sub() { name=$1; shift; .venv/bin/python -m jevdrive.cl submit --name "lat1-$name" --log-dir "$L/pool_$name" "$@" | tail -1; }
run=$(ls -d "$R"/c0/sh30_400_rerun/*/ | tail -1); ls "${run}rollouts" > "$L/scenes400.txt"
m=$(sub msgs --vram 0.5 --cpu 6 --ram 16 -- bash -c "rm -rf $L/msgs400.tmp; $DATA_DIR/third_party/alpasim/.venv/bin/python $here/c1_extract.py msgs \
--run ${run%/} --scenes $L/scenes400.txt --out $L/msgs400.tmp --jobs 6 && rm -rf $L/msgs400 && mv $L/msgs400.tmp $L/msgs400 && touch $L/msgs.DONE")
e=$(sub equiv --after "$m" --vram 9 --cpu 4 --ram 16 -- bash -c "$env $PY $C remap --msgs $L/msgs400 --out $L/remap.json --n 100 && \
$env $PY $C prof --msgs $L/msgs400 --out $L/prof.json && $env $PY $C equiv --msgs $L/msgs400 --out $L/equiv.json && touch $L/equiv.DONE")
ld=""; for s in 8 2; do for k in cpu gpu; do ld+="$env $PY $C load --msgs $L/msgs400 --out $L/load_${k}_s$s.json --synth $k --streams $s --n 64 && "; done; done
l=$(sub load --after "$m" --vram 6 --cpu 8 --ram 16 -- bash -c "$ld $PY $C same --msgs $L/load_cpu_s8.json --out $L/load_gpu_s8.json > $L/load_same.json && touch $L/load.DONE")
ov="+e2e_challenge_nuplan=full runtime.nr_workers=2 runtime.endpoints.renderer.n_concurrent_rollouts=8 runtime.endpoints.driver.n_concurrent_rollouts=8 \
runtime.endpoints.controller.n_concurrent_rollouts=8 defines.nre_cache_size=9"
prev=""; for k in gpu cpu; do
  D=$L/runs/s1-$k/$(date +%Y%m%d-%H%M%S); mkdir -p "$D"
  prev=$(sub "loop-$k" ${prev:+--after "$prev"} --vram 32 --cpu 8 --ram 25 --timeout-h 3 --tries 2 -- bash -c \
"env SH30_TAG=$tag SH30_SYNTH=$k bash experiments/alpasim/scripts/run.sh $D sh30 --scene-list $R/m1/lists/s1.txt $ov && touch $L/loop-$k.DONE")
done
echo "msgs $m equiv $e load $l loop(cpu) $prev -> $L" | tee "$L/CHAIN"
