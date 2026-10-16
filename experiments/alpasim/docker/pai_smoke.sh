#!/usr/bin/env bash
# PAI-track smoke of a submission image in the official containerised stack, AlpaSim unmodified at the deployed commit (the PAI twin of
# smoke.sh; the stack is the one of scripts/pai_run.sh):
#   1. the driver container exactly as the organisers start it: the image's own CMD (serve.py), no mounts, no command, only the replica
#      variables (plus DRV_ENV when given), the limits of test.sh, on a Docker network without a route out (`--internal`), own GPU
#   2. the wizard inside the trusted image `alpasim-base:<version>` with `+e2e_challenge=dev` and wizard.run_method=NONE, the dev preset's
#      `--enable-harmonizer` removed from the renderer command (the leaderboard's ec2 preset renders unharmonised)
#   3. `docker compose up --exit-code-from runtime-0`: NRE renderer, physics, controller, runtime + eval
#   4. compare.py: scene scores against a native reference run of the same scene list (a box pai_box.sh chunk: the simulator repeats a
#      scene only within an identical list) and the driver's latency log
#   experiments/alpasim/docker/pai_smoke.sh <image:version> <out dir> <scene list> [reference results-summary.json]
# Env: DATA_DIR (/data), ALPASIM_SRC, NUREC (scene cache with all-usdzs/), BASE (trusted image), GPU (card of renderer + physics + driver,
#   default 1), CONC (concurrent rollouts, default 4), WIZ (wizard overrides, space separated, e.g.
#   "controller.gains.rel_front_steering_angle_weight=10"), DRV_ENV ("K=V ..." for the driver; empty = the image's baked defaults, which is
#   what a submission serves), PAI_BASEPORT (default 6700; pai_eval.sh uses 6400 / 6500).
# Writes <out>/{STATUS, DONE | ERROR, log.txt, cmd.txt, wizard_cmd.txt, smoke.json, compare.md, drive.jsonl, driver.log, driver-logs/, run.json}
# and the run under <out>/sim (aggregate/results-summary.json, rollouts/, txt-logs/). Mp4s are dropped at the end.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
IMG=$1 OUT=$(mkdir -p "$2" && cd "$2" && pwd) LIST=$3 REF=${4:-}
D=${DATA_DIR:-/data}; SRC=${ALPASIM_SRC:-$D/third_party/alpasim}; NUREC=${NUREC:-$D/datasets/nurec}
BASE=${BASE:-alpasim-base:$(sed -n 's/^version = "\(.*\)"/\1/p' "$SRC/pyproject.toml" | head -1)}
GPU=${GPU:-1}; CONC=${CONC:-4}; PORT0=${PAI_BASEPORT:-6700}
NAME=jevpai-$(basename "$OUT" | tr -c 'a-zA-Z0-9\n' '-' | tr 'A-Z' 'a-z'); NET=pai-nonet; N=$NAME-drv; R=$OUT/sim
rm -f "$OUT"/{DONE,ERROR,.stop}; mkdir -p "$R"; exec > >(tee -a "$OUT/log.txt") 2>&1
st() { echo "$(date '+%F %T') $*" | tee "$OUT/STATUS"; }
stop_all() { [[ -f $R/docker-compose.yaml ]] && docker compose -p "$NAME" -f "$R/docker-compose.yaml" down >/dev/null 2>&1
             docker logs "$N" > "$OUT/driver.log" 2>&1; docker rm -f "$N" >/dev/null 2>&1; touch "$OUT/.stop"; }
die() { st "ERROR $*"; echo "$*" > "$OUT/ERROR"; stop_all; exit 1; }
for i in "$IMG" "$BASE"; do docker image inspect "$i" >/dev/null 2>&1 || die "image $i missing"; done
docker network inspect $NET >/dev/null 2>&1 || docker network create --internal $NET >/dev/null || die "network"
ids=$(grep . "$LIST"); n=$(echo "$ids" | wc -l)

st "driver $IMG (its own CMD${DRV_ENV:+, env $DRV_ENV}) on GPU $GPU"
run=(docker run -d --name "$N" --init --cap-drop ALL --security-opt no-new-privileges:true --read-only --pids-limit 1024 --memory 32g --cpus 8
     --tmpfs /tmp:rw,nosuid,nodev,size=2g,mode=1777 --tmpfs /run:rw,nosuid,nodev,size=64m,mode=0755 --network $NET --gpus "device=$GPU"
     -e ALPASIM_CONTESTANT_REPLICA_INDEX=0 -e ALPASIM_CONTESTANT_REPLICAS=1)
for kv in ${DRV_ENV:-}; do run+=(-e "$kv"); done
printf '%q ' "${run[@]}" "$IMG" > "$OUT/cmd.txt"; echo >> "$OUT/cmd.txt"
t0=$(date +%s); "${run[@]}" "$IMG" >/dev/null || die "docker run"
until docker logs "$N" 2>&1 | grep -q "listening on"; do
  [[ $(docker inspect -f '{{.State.Running}}' "$N") == true ]] || die "driver exited before listening"; (( $(date +%s) - t0 > 600 )) && die "driver not listening"; sleep 0.5
done
ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$N")
docker exec "$N" python -c "import urllib.request as u; u.urlopen('https://pypi.org', timeout=8)" >/dev/null 2>&1 && die "the driver container reached the internet"
( peak=0; tmp=0; while [[ ! -f $OUT/.stop ]]; do
    u=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | while IFS=', ' read -r pid mem; do
          grep -q "$(docker inspect -f '{{.Id}}' "$N" 2>/dev/null | cut -c1-12)" "/proc/$pid/cgroup" 2>/dev/null && echo "$mem"; done | paste -sd+ | bc)
    (( ${u:-0} > peak )) && peak=$u
    a=$(docker exec "$N" sh -c 'df -B1 --output=used /tmp | tail -1' 2>/dev/null); (( ${a:-0} > tmp )) && tmp=$a
    echo "$peak $tmp" > "$OUT/.peak"; sleep 2; done ) & sampler=$!

st "wizard in $BASE (run_method NONE), driver at $ip:6789, $n scenes, $CONC concurrent, GPU $GPU"
wiz=(uv run alpasim_wizard +e2e_challenge=dev "scenes.scene_ids=[$(echo "$ids" | sed 's/.*/"&"/' | paste -sd,)]" scenes.limit_to_first_n=0
     "scenes.scene_cache=$NUREC" "scenes.scenes_csv=[$SRC/data/scenes/sim_scenes.csv,$SRC/data/scenes/sim_scenes_2604.csv]"
     "services.renderer.gpus=[$GPU]" "services.physics.gpus=[$GPU]" "wizard.baseport=$PORT0"
     "runtime.endpoints.renderer.n_concurrent_rollouts=$CONC" "runtime.endpoints.driver.n_concurrent_rollouts=$CONC"
     "runtime.endpoints.physics.n_concurrent_rollouts=$CONC" "runtime.endpoints.controller.n_concurrent_rollouts=$CONC"
     "defines.nre_cache_size=$((CONC + 1))" wizard.run_method=NONE "wizard.log_dir=$R" ${WIZ:-})
printf '%q ' "${wiz[@]}" > "$OUT/wizard_cmd.txt"; echo >> "$OUT/wizard_cmd.txt"
docker run --rm --gpus all -v "$SRC:$SRC" -v "$R:$R" -v "$NUREC:$NUREC" -e "PYTHONPATH=$SRC/src/wizard:$SRC/src/utils" -e HF_HUB_OFFLINE=1 \
  -e "ALPASIM_DRIVER_HOST=$ip" -e ALPASIM_DRIVER_PORT=6789 "$BASE" \
  bash -c 'umask 0000; git config --global --add safe.directory "*"; cd /repo && "$@"' wizard "${wiz[@]}" > "$OUT/wizard.log" 2>&1 || die "wizard failed, see wizard.log"
[[ -f $R/docker-compose.yaml ]] || die "no docker-compose.yaml"
python3 - "$R/docker-compose.yaml" <<'PY' || die "compose patch"
import sys, yaml                                    # the ec2 preset's renderer: no harmonizer (dev preset as shipped enables it)
f = sys.argv[1]; d = yaml.safe_load(open(f))
for k, s in d["services"].items():
    if k.startswith("renderer"):
        s["command"] = [c.replace(" --enable-harmonizer", "") for c in s["command"]]
yaml.safe_dump(d, open(f, "w"), sort_keys=False)
PY

st "docker compose up ($NAME)"
t1=$(date +%s)
(cd "$R" && docker compose -p "$NAME" -f docker-compose.yaml up --exit-code-from runtime-0) > "$OUT/compose.log" 2>&1; rc=$?
wall=$(( $(date +%s) - t1 ))
docker exec "$N" tar -C /tmp/alpasim-driver -c . 2>/dev/null | { mkdir -p "$OUT/driver-logs"; tar -x -C "$OUT/driver-logs"; }   # /tmp is a tmpfs: no docker cp
cp "$OUT/driver-logs/drive.jsonl" "$OUT/drive.jsonl" 2>/dev/null
read -r peak tmp < "$OUT/.peak"; stop_all; wait "$sampler"; rm -f "$OUT"/.stop "$OUT"/.peak
find "$R/rollouts" -name '*.mp4' -delete 2>/dev/null
(( rc == 0 )) || die "compose exited with $rc after $wall s, see compose.log"
[[ -f $R/aggregate/results-summary.json ]] || die "no results-summary.json"
ver=$(grep -m1 -o 'serve: driver.*' "$OUT/driver.log")
echo "{\"scenes\": $n, \"conc\": $CONC, \"wall_s\": $wall, \"gpu\": $GPU, \"image\": \"$IMG\", \"base\": \"$BASE\", \"wiz\": \"${WIZ:-}\", \"drv_env\": \"${DRV_ENV:-}\"}" > "$OUT/run.json"
python3 "$here/compare.py" "$R/aggregate/results-summary.json" ${REF:+--ref "$REF"} $( [[ -s $OUT/drive.jsonl ]] && echo "--drive $OUT/drive.jsonl" ) \
  --label "$(basename "$OUT")" --meta "image=$IMG" "base=$BASE" "wall_s=$wall" "driver_gpu_peak_mib=$peak" "tmp_peak_bytes=$tmp" "serve=$ver" "wiz=${WIZ:-}" \
  --out "$OUT" || die "compare"
date > "$OUT/DONE"; st "done: $(sed -n 3p "$OUT/compare.md")"
