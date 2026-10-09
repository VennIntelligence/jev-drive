#!/usr/bin/env bash
# End-to-end smoke of a submission image in the official containerised stack, AlpaSim unmodified at the deployed commit:
#   1. the driver container, with the organisers' limits (test.sh) on a Docker network without a route out (`--internal`), own GPU
#   2. the wizard, run inside the trusted image `alpasim-base:<version>` (root Dockerfile, built as shipped) with wizard.run_method=NONE:
#      it writes docker-compose.yaml and the service configs (the two-step form of src/wizard/configs/e2e_challenge/README.md)
#   3. `docker compose up --exit-code-from runtime-0`: MTGS renderer, controller, runtime + eval in containers of that image
#   4. compare.py: scene scores against a native reference run (results-summary.json of run_native.py on the GPU box)
# The wizard overrides are those of the native reference runs (docs/alpasim.md), so the scene list and concurrency match.
#   experiments/alpasim/docker/smoke.sh <image:version> <out dir> [JEV_TAG] [reference results-summary.json]
# Env: DATA_DIR, ALPASIM_SRC, NUPLAN_ROOT, SCENES (scene list file), BASE (trusted image), DGPU (driver card, default 1; the stack's
# 1gpu topology renders on card 0), WIZ_EXTRA (more wizard overrides), DRV_ENV ("K=V ..." for the driver, e.g. SH30_DUMP=16). Writes <out>/{STATUS, DONE | ERROR, log.txt, smoke.json, compare.md, drive.jsonl, driver.log} and the
# run itself under <out>/run (aggregate/results-summary.json, rollouts/, txt-logs/), the driver's own logs under <out>/driver-logs.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
IMG=$1 OUT=$(mkdir -p "$2" && cd "$2" && pwd) TAG=${3:-} REF=${4:-}
: "${DATA_DIR:?DATA_DIR}"
SRC=${ALPASIM_SRC:-$DATA_DIR/third_party/alpasim}
NUPLAN=${NUPLAN_ROOT:-$DATA_DIR/datasets/alpasim_nuplan}
SCENES=${SCENES:-$DATA_DIR/runs/alpasim/scenes_navtest_full_part001_48.txt}
BASE=${BASE:-alpasim-base:$(sed -n 's/^version = "\(.*\)"/\1/p' "$SRC/pyproject.toml" | head -1)}
DGPU=${DGPU:-1} NET=jev-nonet N=jev-drv-smoke-$$ R=$OUT/run
rm -f "$OUT"/{DONE,ERROR}; mkdir -p "$R"; exec > >(tee -a "$OUT/log.txt") 2>&1
st() { echo "$(date '+%F %T') $*" | tee "$OUT/STATUS"; }
stop_all() { [[ -f $R/docker-compose.yaml ]] && docker compose -f "$R/docker-compose.yaml" down --remove-orphans >/dev/null 2>&1
             docker logs "$N" > "$OUT/driver.log" 2>&1; docker rm -f "$N" >/dev/null 2>&1; touch "$OUT/.stop"; }
die() { st "ERROR $*"; echo "$*" > "$OUT/ERROR"; stop_all; exit 1; }
docker image inspect "$BASE" >/dev/null 2>&1 || die "trusted image $BASE missing: docker build -t $BASE $SRC"
docker network inspect $NET >/dev/null 2>&1 || docker network create --internal $NET >/dev/null || die "network"

st "driver $IMG ${TAG:-<image default>} on GPU $DGPU"
run=(docker run -d --name "$N" --init --cap-drop ALL --security-opt no-new-privileges:true --read-only --pids-limit 1024 --memory 32g --cpus 8
     --tmpfs /tmp:rw,nosuid,nodev,size=2g,mode=1777 --tmpfs /run:rw,nosuid,nodev,size=64m,mode=0755 --network $NET --gpus "device=$DGPU"
     -e ALPASIM_CONTESTANT_REPLICA_INDEX=0 -e ALPASIM_CONTESTANT_REPLICAS=1)
[[ -n $TAG ]] && run+=(-e "JEV_TAG=$TAG")
for kv in ${DRV_ENV:-}; do run+=(-e "$kv"); done
printf '%q ' "${run[@]}" "$IMG" > "$OUT/cmd.txt"; echo >> "$OUT/cmd.txt"
t0=$(date +%s); "${run[@]}" "$IMG" >/dev/null || die "docker run"
until docker logs "$N" 2>&1 | grep -q "listening on"; do
  [[ $(docker inspect -f '{{.State.Running}}' "$N") == true ]] || die "driver exited before listening"; (( $(date +%s) - t0 > 600 )) && die "driver not listening"; sleep 0.5
done
ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$N")
docker exec "$N" python -c "import urllib.request as u; u.urlopen('https://pypi.org', timeout=8)" >/dev/null 2>&1 && die "the driver container reached the internet"
( peak=0; tmp=0; while [[ ! -f $OUT/.stop ]]; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$DGPU"); (( u > peak )) && peak=$u
    a=$(docker exec "$N" sh -c 'df -B1 --output=used /tmp | tail -1' 2>/dev/null); (( ${a:-0} > tmp )) && tmp=$a
    echo "$peak $tmp" > "$OUT/.peak"; sleep 1; done ) & sampler=$!

# The shipped config builds gsplat's CUDA kernels for sm_89 / sm_90 + PTX (TORCH_CUDA_ARCH_LIST in e2e_challenge_nuplan_common/base.yaml):
# an older render card (RTX 3090 = sm_86) has "no kernel image". There, and only there, the list is overridden to the card's own arch.
RGPU=${RGPU:-0}
cap=$(docker run --rm --gpus "device=$RGPU" --entrypoint python "$IMG" -c "import torch; print('%d.%d' % torch.cuda.get_device_capability(0))") || die "render card capability"
arch=(); [[ $(echo "$cap < 8.9" | bc) == 1 ]] && { arch=("services.renderer.environments=[\"UV_LINK_MODE=copy\",\"TORCH_CUDA_ARCH_LIST=$cap\"]"); echo "DEVIATION: render card is sm_$cap, gsplat kernels built for $cap instead of 8.9;9.0+PTX"; }
st "wizard in $BASE (run_method NONE), driver at $ip:6789, $(grep -c . "$SCENES") scenes"
ids=$(grep . "$SCENES" | sed 's/.*/"&"/' | paste -sd,)
wiz=(uv run alpasim_wizard +e2e_challenge_nuplan=full runtime.nr_workers=2 runtime.endpoints.renderer.n_concurrent_rollouts=8
     runtime.endpoints.driver.n_concurrent_rollouts=8 runtime.endpoints.controller.n_concurrent_rollouts=8 defines.nre_cache_size=9
     "scenes.scene_ids=[$ids]" scenes.limit_to_first_n=0 wizard.run_method=NONE "wizard.log_dir=$R" "${arch[@]}" ${WIZ_EXTRA:-})
printf '%q ' "${wiz[@]}" > "$OUT/wizard_cmd.txt"; echo >> "$OUT/wizard_cmd.txt"
# PYTHONPATH puts the checkout's wizard before the image's own copy of the same commit, so repo-relative mounts are host paths.
docker run --rm --gpus all -v "$SRC:$SRC" -v "$R:$R" -v "$NUPLAN:$NUPLAN:ro" -e "PYTHONPATH=$SRC/src/wizard:$SRC/src/utils" \
  -e "ALPASIM_DRIVER_HOST=$ip" -e ALPASIM_DRIVER_PORT=6789 -e "ALPASIM_NUPLAN_ROOT=$NUPLAN" "$BASE" \
  bash -c 'umask 0000; git config --global --add safe.directory "*"; cd /repo && "$@"' wizard "${wiz[@]}" > "$OUT/wizard.log" 2>&1 || die "wizard failed, see wizard.log"
[[ -f $R/docker-compose.yaml ]] || die "no docker-compose.yaml"

st "docker compose up"
t1=$(date +%s)
(cd "$R" && docker compose -f docker-compose.yaml up --remove-orphans --exit-code-from runtime-0) > "$OUT/compose.log" 2>&1; rc=$?
wall=$(( $(date +%s) - t1 ))
docker exec "$N" tar -C /tmp/alpasim-driver -c . 2>/dev/null | { mkdir -p "$OUT/driver-logs"; tar -x -C "$OUT/driver-logs"; }   # /tmp is a tmpfs: no docker cp
cp "$OUT/driver-logs/drive.jsonl" "$OUT/drive.jsonl" 2>/dev/null
read -r peak tmp < "$OUT/.peak"; stop_all; wait "$sampler"; rm -f "$OUT"/.stop "$OUT"/.peak
(( rc == 0 )) || die "compose exited with $rc after $wall s, see compose.log"
[[ -f $R/aggregate/results-summary.json ]] || die "no results-summary.json"
python3 "$here/compare.py" "$R/aggregate/results-summary.json" ${REF:+--ref "$REF"} --drive "$OUT/drive.jsonl" --label "${TAG:-default}" \
  --meta "image=$IMG" "base=$BASE" "render_arch=$cap" "wall_s=$wall" "driver_gpu_peak_mib=$peak" "tmp_peak_bytes=$tmp" --out "$OUT" || die "compare"
date > "$OUT/DONE"; st "done: $(head -3 "$OUT/compare.md" | tail -1)"
