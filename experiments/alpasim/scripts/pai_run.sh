#!/usr/bin/env bash
# One PAI-track closed-loop run on the Tokyo box: the official containerised stack (`+e2e_challenge=dev`: NRE renderer, physics,
# nonlinear MPC, runtime + eval; AlpaSim unmodified at the deployed commit) against our driver in the nuPlan submission image with
# experiments/alpasim/lib/pai_{core,driver}.py mounted over it. Modelled on docker/smoke.sh; shares the box with it, so everything
# here has its own names: compose project pai-<name>, network pai-nonet, ports from PAI_BASEPORT, containers pai-*.
#   pai_run.sh <out dir> <scenes.tsv | file of scene ids> [wizard overrides...]
# Env: DATA_DIR (/data), ALPASIM_SRC, NUREC (scene cache with all-usdzs/, pai_fetch.sh), IMG (driver image), BASE (trusted image),
#   GPU (card of renderer + physics, default 1), DGPU (driver card, default $GPU), STAGE (rows of the tsv with stage <= this, default 1),
#   CONC (concurrent rollouts, default 1), TAG (checkpoint, default P2H10-F-s0), DRV_ENV (extra `-e K=V` for the driver, space
#   separated), DRV_PY (a driver variant in lib/ run instead of pai_driver.py, e.g. col1_pai_driver.py), PAI_BASEPORT (default 6400), HARMONIZER (0 = the renderer of the public leaderboard: the dev preset's
#   `--enable-harmonizer` removed, as the ec2 preset does; 1 = the dev preset as shipped, which is how the bundled reference runs were
#   made; the weights are then mounted from $NUREC/harmonizer because the renderer cannot fetch them from inside the container).
# Writes <out>/{STATUS, DONE | ERROR, log.txt, usage.jsonl (2 s: per-process VRAM, per-container RAM / CPU), driver/ (drive.jsonl,
# start.jsonl, dump/), sim/ (the run: aggregate/results-summary.json, rollouts/, txt-logs/)}.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
OUT=$(mkdir -p "$1" && cd "$1" && pwd) LIST=$2; shift 2
D=${DATA_DIR:-/data}; SRC=${ALPASIM_SRC:-$D/third_party/alpasim}; NUREC=${NUREC:-$D/datasets/nurec}
IMG=${IMG:-jev-alpasim:p2h10-f-s0-6f3d05d1}
BASE=${BASE:-alpasim-base:$(sed -n 's/^version = "\(.*\)"/\1/p' "$SRC/pyproject.toml" | head -1)}
GPU=${GPU:-1}; DGPU=${DGPU:-$GPU}; STAGE=${STAGE:-1}; CONC=${CONC:-1}; TAG=${TAG:-P2H10-F-s0}; PORT0=${PAI_BASEPORT:-6400}
NAME=pai-$(basename "$OUT" | tr -c 'a-zA-Z0-9\n' '-'); NET=pai-nonet; N=$NAME-drv; R=$OUT/sim
rm -f "$OUT"/{DONE,ERROR,.stop}; mkdir -p "$R" "$OUT/driver"; chmod 777 "$OUT/driver"; exec > >(tee -a "$OUT/log.txt") 2>&1
st() { echo "$(date '+%F %T') $*" | tee "$OUT/STATUS"; }
stop_all() { [[ -f $R/docker-compose.yaml ]] && docker compose -p "$NAME" -f "$R/docker-compose.yaml" down >/dev/null 2>&1
             docker logs "$N" > "$OUT/driver.log" 2>&1; docker rm -f "$N" >/dev/null 2>&1; touch "$OUT/.stop"; }
die() { st "ERROR $*"; echo "$*" > "$OUT/ERROR"; stop_all; exit 1; }
for i in "$IMG" "$BASE"; do docker image inspect "$i" >/dev/null 2>&1 || die "image $i missing"; done
docker network inspect $NET >/dev/null 2>&1 || docker network create --internal $NET >/dev/null || die "network"
if head -1 "$LIST" | grep -q '^scene_id'; then ids=$(awk -F'\t' -v s="$STAGE" 'NR > 1 && $8 <= s {print $1}' "$LIST"); else ids=$(grep . "$LIST"); fi
n=$(echo "$ids" | wc -l)

st "driver $IMG $TAG on GPU $DGPU"
lib=$here/../lib
# shellcheck disable=SC2086
docker run -d --name "$N" --init --cap-drop ALL --security-opt no-new-privileges:true --read-only --pids-limit 1024 --memory 32g --cpus 8 \
  --tmpfs /tmp:rw,nosuid,nodev,size=2g,mode=1777 --tmpfs /run:rw,nosuid,nodev,size=64m,mode=0755 --network $NET --gpus "device=$DGPU" \
  -v "$lib/pai_core.py:/app/jev-drive/experiments/alpasim/lib/pai_core.py:ro" -v "$lib/pai_driver.py:/app/jev-drive/experiments/alpasim/lib/pai_driver.py:ro" \
  ${DRV_PY:+-v "$lib/$DRV_PY:/app/jev-drive/experiments/alpasim/lib/$DRV_PY:ro"} \
  -v "$OUT/driver:/logs" -e ALPASIM_DRIVER_LOG_DIR=/logs -e "SH30_TAG=$TAG" ${DRV_ENV:-} \
  "$IMG" python "/app/jev-drive/experiments/alpasim/lib/${DRV_PY:-pai_driver.py}" >/dev/null || die "docker run"
t0=$(date +%s)
until docker logs "$N" 2>&1 | grep -q "listening on"; do
  [[ $(docker inspect -f '{{.State.Running}}' "$N") == true ]] || die "driver exited before listening"; (( $(date +%s) - t0 > 600 )) && die "driver not listening"; sleep 0.5
done
ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$N")
( while [[ ! -f $OUT/.stop ]]; do
    python3 - "$NAME" >> "$OUT/usage.jsonl" <<'EOF'
import json, subprocess, sys, time
run = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout
gpu = [l.split(", ") for l in run("nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader,nounits").splitlines()]
own = {}
for pid, mem, uuid in gpu:                                  # a card process belongs to the container named in its cgroup's docker id
    cid = run(f"grep -o -m1 'docker-[0-9a-f]*' /proc/{pid}/cgroup").strip()[7:19]
    name = run(f"docker ps --filter id={cid} --format '{{{{.Names}}}}'").strip() if cid else ""
    if name.startswith(sys.argv[1]):
        own[name] = own.get(name, 0) + int(mem)
st = [l.split("|") for l in run("docker stats --no-stream --format '{{.Name}}|{{.MemUsage}}|{{.CPUPerc}}'").splitlines() if l.startswith(sys.argv[1])]
print(json.dumps({"t": time.time(), "vram_mib": own, "ram": {a: b.split(" / ")[0] for a, b, _ in st}, "cpu": {a: c for a, _, c in st},
                  "cards_mib": run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").split()}))
EOF
  done ) & sampler=$!

st "wizard in $BASE (run_method NONE), driver at $ip:6789, $n scenes, $CONC concurrent, render GPU $GPU"
wiz=(uv run alpasim_wizard +e2e_challenge=dev "scenes.scene_ids=[$(echo "$ids" | sed 's/.*/"&"/' | paste -sd,)]" scenes.limit_to_first_n=0
     "scenes.scene_cache=$NUREC" "scenes.scenes_csv=[$SRC/data/scenes/sim_scenes.csv,$SRC/data/scenes/sim_scenes_2604.csv]"
     "services.renderer.gpus=[$GPU]" "services.physics.gpus=[$GPU]" "wizard.baseport=$PORT0"
     "runtime.endpoints.renderer.n_concurrent_rollouts=$CONC" "runtime.endpoints.driver.n_concurrent_rollouts=$CONC"
     "runtime.endpoints.physics.n_concurrent_rollouts=$CONC" "runtime.endpoints.controller.n_concurrent_rollouts=$CONC"
     "defines.nre_cache_size=$((CONC + 1))" wizard.run_method=NONE "wizard.log_dir=$R" "$@")
printf '%q ' "${wiz[@]}" > "$OUT/wizard_cmd.txt"; echo >> "$OUT/wizard_cmd.txt"
docker run --rm --gpus all -v "$SRC:$SRC" -v "$R:$R" -v "$NUREC:$NUREC" -e "PYTHONPATH=$SRC/src/wizard:$SRC/src/utils" -e HF_HUB_OFFLINE=1 \
  -e "ALPASIM_DRIVER_HOST=$ip" -e ALPASIM_DRIVER_PORT=6789 "$BASE" \
  bash -c 'umask 0000; git config --global --add safe.directory "*"; cd /repo && "$@"' wizard "${wiz[@]}" > "$OUT/wizard.log" 2>&1 || die "wizard failed, see wizard.log"
[[ -f $R/docker-compose.yaml ]] || die "no docker-compose.yaml"
python3 - "$R/docker-compose.yaml" "${HARMONIZER:-0}" "$NUREC/harmonizer" <<'PY' || die "compose patch"
import sys, yaml
f, harm, cache = sys.argv[1:]
d = yaml.safe_load(open(f))
for k, s in d["services"].items():
    if k.startswith("renderer"):
        if harm == "1":
            s["volumes"].append(f"{cache}:/home/.cache/nre/harmonizer")
        else:
            s["command"] = [c.replace(" --enable-harmonizer", "") for c in s["command"]]
yaml.safe_dump(d, open(f, "w"), sort_keys=False)
PY

st "docker compose up ($NAME)"
t1=$(date +%s)
(cd "$R" && docker compose -p "$NAME" -f docker-compose.yaml up --exit-code-from runtime-0) > "$OUT/compose.log" 2>&1; rc=$?
wall=$(( $(date +%s) - t1 ))
stop_all; wait "$sampler"; rm -f "$OUT/.stop"
(( rc == 0 )) || die "compose exited with $rc after $wall s, see compose.log"
[[ -f $R/aggregate/results-summary.json ]] || die "no results-summary.json"
echo "{\"scenes\": $n, \"conc\": $CONC, \"wall_s\": $wall, \"gpu\": $GPU, \"driver_gpu\": $DGPU, \"tag\": \"$TAG\", \"harmonizer\": ${HARMONIZER:-0}, \"image\": \"$IMG\", \"base\": \"$BASE\"}" > "$OUT/run.json"
date > "$OUT/DONE"; st "done: $n scenes in $wall s"
