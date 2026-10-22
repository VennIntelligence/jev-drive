#!/usr/bin/env bash
# A human drives AlpaSim with the phone wheel (Tokyo box, card 0): the official containerised stack (AlpaSim unmodified at the deployed
# commit; renderer / physics / controller / runtime + eval in their images) against lib/human_driver.py, sequential scenes, one at a time.
# Called by human_pai.sh / human_nuplan.sh:   human_stack.sh pai|nuplan <out dir> <scenes> [wizard overrides...]
#   scenes: PAI = a pai_scenes.py tsv (rows with stage <= STAGE, default 1) or a file of scene ids; nuPlan = a file of scene ids.
# Starts tmux session `human`: window alpa-viewer (scripts/human_viewer.py on DISPLAY=:0) and alpa-drive (this script with --run).
# Rendering (README "Speed"): HUMAN_MODE=fast (default: front camera only at HUMAN_H px height, no eval video, a human experience,
#   not a leaderboard number) | official (the presets' cameras and eval as shipped). Knobs under fast: HUMAN_CAMS (front | all),
#   HUMAN_H (render height px; 1080, 288 with the harmonizer), HUMAN_FI_MS (camera frame interval ms; PAI 100, 200 with the harmonizer;
#   nuPlan 500), HUMAN_VIDEO (0 | 1), HUMAN_NRE (default | nrend: PAI renderer
#   backend), HUMAN_PHYS (1 | 0: PAI ground physics). HUMAN_HARMONIZER (0 | 1, PAI only, both modes: keep the dev preset's
#   --enable-harmonizer and mount the weights from $NUREC/harmonizer, as pai_run.sh HARMONIZER=1; default 0 = unharmonised, as the
#   public leaderboard). With it the defaults are HUMAN_H 288, HUMAN_FI_MS 200 and HUMAN_HARM_RES "288 512" (the renderer's shipped
#   --harmonizer-resolution flag; shipped "576 1024", HUMAN_HARM_RES= keeps it): real time, wall / sim 0.75 unpaced vs 3.7 as shipped. Other env: HUMAN_GPU (0; card 1 is refused unless HUMAN_ALLOW_CARD1=1),
#   HUMAN_PREFIX (hum: container / compose names; a probe beside the user's stack uses another), HUMAN_JIT_DIR, HUMAN_TRACK_GATE (1: the
#   plan goes through track_ref's gate, lib/human_driver.py; 0 = the raw plan), HUMAN_BASEPORT (6700: renderer,
#   physics, controller), HUMAN_DRV_PORT (6790), HUMAN_START (click | throttle | none), HUMAN_WHEEL_PORT (47800; 47801 = fake_wheel.py),
#   HUMAN_REALTIME (1.0: the driver paces sim time to wall time; 0 = as fast as the stack runs), HUMAN_MODEL_REALTIME (0: model mode's sessions unpaced; 1.0 = paced as the human), HUMAN_LIVE (/data/runs/alpasim_human/live), HUMAN_VIEWER (1), HUMAN_IMG (image the driver runs in), HUMAN_MIN_FREE_MB (8000), STAGE,
#   HUMAN_FG (0; 1 = run in the foreground after the checks instead of tmux human:alpa-drive: scripts/human_loop.py),
#   HUMAN_CONTROL (the desktop panel's control.json: pause; lib/human_driver.py),
#   HUMAN_GT (1: the runtime sends the recorded log ego trajectory to the driver every step, send_recording_ground_truth, for the
#   viewer's shadow; display only, scoring unchanged; 0 = as shipped). HUMAN_FAST_ABORT (default = HUMAN_SERVE): a skipped scene's
#   next drive call fails, so the runtime ends that rollout at once (unscored) instead of running it out (lib/human_driver.py).
# HUMAN_SERVE=1 (scripts/human_loop.py): a resident stack. The runtime runs in AlpaSim's shipped daemon mode (wizard.run_mode=SERVER,
#   runtime port HUMAN_BASEPORT + 4); after `compose up -d` the services and the driver stay up and scripts/human_feed.py (in the
#   driver image) sends one simulate request per scene from <out>/feed/req/ until <out>/feed/stop, SIGTERM, a dead service or
#   HUMAN_IDLE_S (1200) with no client. <scenes> = every scene the stack may serve. Progress: <out>/phase.json {phase, msg, t0, t}.
# Writes <out>/{STATUS, DONE | ERROR, log.txt, run.json, usage.jsonl (card VRAM, 2 s), driver/{drive,start}.jsonl, driver.log,
#   sim/ (aggregate/results-summary.json, rollouts/, txt-logs/), results.md / results.json (human_results.py)}.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$here/../../.." && pwd)
TRACK=$1; shift
D=${DATA_DIR:-/data}; PFX=${HUMAN_PREFIX:-hum}; LIVE=${HUMAN_LIVE:-$D/runs/alpasim_human/live}; GPU=${HUMAN_GPU:-0}
export DISPLAY=:0 XAUTHORITY=${XAUTHORITY:-/run/user/1001/gdm/Xauthority}
launcher() { mkdir -p "$LIVE"; python3 -c 'import json,sys,time; print(json.dumps(dict(state=sys.argv[1], msg=sys.argv[2], run=sys.argv[3], t=time.time())))' "$@" > "$LIVE/launcher.tmp" && mv "$LIVE/launcher.tmp" "$LIVE/launcher.json"; }

if [[ ${1:-} != --run ]]; then                                    # outer: checks, viewer, then the run in tmux
  [[ $TRACK == pai || $TRACK == nuplan ]] || { echo "track pai | nuplan"; exit 1; }
  OUT=$(mkdir -p "$1" && cd "$1" && pwd); LIST=$(cd "$(dirname "$2")" && pwd)/$(basename "$2"); shift 2
  [[ $GPU == 1 && ${HUMAN_ALLOW_CARD1:-0} != 1 ]] && { echo "card 1 runs the PAI chain: refusing HUMAN_GPU=1 (HUMAN_ALLOW_CARD1=1 when it is idle)"; exit 1; }
  free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
  uuid=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
  nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader | grep "^$uuid" | grep -q CarlaUE4 && {
    echo "card $GPU runs CARLA (a B2D exam?)"; launcher refused "card $GPU runs CARLA" ""; exit 1; }
  (( free < ${HUMAN_MIN_FREE_MB:-8000} )) && { echo "card $GPU has $free MiB free"; launcher refused "card $GPU has only $free MiB free" ""; exit 1; }
  P0=${HUMAN_BASEPORT:-6700}
  for p in $P0 $((P0 + 1)) $((P0 + 2)) $((P0 + 3)) $((P0 + 4)) ${HUMAN_DRV_PORT:-6790}; do
    ss -ltn | grep -q ":$p " && { echo "port $p is taken (another human run?)"; launcher refused "port $p taken" ""; exit 1; }
  done
  docker ps --format '{{.Names}}' | grep -q "^$PFX-" && { echo "a $PFX-* stack is running"; docker ps --format '{{.Names}}' | grep "^$PFX-"; exit 1; }
  tmux has-session -t human 2>/dev/null || tmux new-session -d -s human -n main
  if [[ ${HUMAN_VIEWER:-1} == 1 ]] && ! tmux list-windows -t human -F '#{window_name}' | grep -qx alpa-viewer; then
    tmux new-window -d -t human -n alpa-viewer "cd $REPO && DISPLAY=:0 XAUTHORITY=$XAUTHORITY SDL_VIDEO_WINDOW_POS=${HUMAN_WIN_POS:-1340,40} \
      /data/envs/carla/bin/python experiments/alpasim_human/scripts/human_viewer.py --dir $LIVE --window-size ${HUMAN_WIN_SIZE:-2480 1736}; exec bash"
  fi
  launcher running "starting the $TRACK stack" "$OUT"
  [[ ${HUMAN_FG:-0} == 1 ]] && exec bash "$here/human_stack.sh" "$TRACK" --run "$OUT" "$LIST" "$@"
  tmux kill-window -t human:alpa-drive 2>/dev/null
  extra=""; (( $# )) && extra=$(printf '%q ' "$@")
  envs=$(env | grep -E '^(HUMAN_[A-Z_]+|STAGE|DATA_DIR|NUREC|NUPLAN_ROOT|ALPASIM_SRC|BASE)=' | while IFS= read -r l; do printf '%q ' "$l"; done)
  tmux new-window -d -t human -n alpa-drive "env $envs bash $here/human_stack.sh $TRACK --run $OUT $LIST $extra; exec bash"
  echo "tmux human:alpa-viewer, human:alpa-drive; out $OUT"
  exit 0
fi

OUT=$2 LIST=$3; shift 3
SRC=${ALPASIM_SRC:-$D/third_party/alpasim}; NUREC=${NUREC:-$D/datasets/nurec}; NUPLAN=${NUPLAN_ROOT:-$D/datasets/alpasim_nuplan}
IMG=${HUMAN_IMG:-jev-alpasim:p2h10-f-s0-6f3d05d1}
BASE=${BASE:-alpasim-base:$(sed -n 's/^version = "\(.*\)"/\1/p' "$SRC/pyproject.toml" | head -1)}
P0=${HUMAN_BASEPORT:-6700}; DPORT=${HUMAN_DRV_PORT:-6790}; MODE=${HUMAN_MODE:-fast}
NAME=$PFX-$(basename "$OUT" | tr -c 'a-zA-Z0-9\n' '-' | tr 'A-Z' 'a-z'); N=$NAME-drv; R=$OUT/sim
SERVE=${HUMAN_SERVE:-0}; RT=$((P0 + 4)); T0=$(date +%s.%N)
rm -f "$OUT"/{DONE,ERROR,.stop}; mkdir -p "$R" "$OUT/driver" "$LIVE"; exec > >(tee -a "$OUT/log.txt") 2>&1; echo $$ > "$OUT/pid"
st() { echo "$(date '+%F %T') $*" | tee "$OUT/STATUS"; }
phase() { python3 -c 'import json,sys,time; print(json.dumps(dict(phase=sys.argv[1], msg=sys.argv[2], t0=float(sys.argv[3]), t=time.time())))' \
            "$1" "$2" "$T0" > "$OUT/.phase.tmp" && mv "$OUT/.phase.tmp" "$OUT/phase.json"; }
stop_all() { docker rm -f "$NAME-feed" >/dev/null 2>&1
             [[ -f $R/docker-compose.yaml ]] && docker compose -p "$NAME" -f "$R/docker-compose.yaml" down -t 5 >/dev/null 2>&1
             docker logs "$N" > "$OUT/driver.log" 2>&1; docker rm -f "$N" >/dev/null 2>&1; touch "$OUT/.stop"; }
die() { st "ERROR $*"; echo "$*" > "$OUT/ERROR"; phase error "$*"; stop_all; [[ $SERVE == 1 ]] || launcher error "$*" "$OUT"; exit 1; }
trap 'die "interrupted"' INT TERM
for i in "$IMG" "$BASE"; do docker image inspect "$i" >/dev/null 2>&1 || die "image $i missing"; done
if head -1 "$LIST" | grep -q '^scene_id'; then ids=$(awk -F'\t' -v s="${STAGE:-1}" 'NR > 1 && $8 <= s {print $1}' "$LIST"); else ids=$(grep . "$LIST"); fi
n=$(echo "$ids" | wc -l); quoted=$(echo "$ids" | sed 's/.*/"&"/' | paste -sd,)

# Rendering levers (README "Speed"). Defaults = the fastest combination that keeps the front image usable.
HARM=0; [[ $TRACK == pai && ${HUMAN_HARMONIZER:-0} == 1 ]] && HARM=1
# Harmonizer on: its cost is per frame at its own resolution (shipped 576 x 1024), not the render height, so the real-time defaults
# are 288 x 512 harmonizer passes on every second 10 Hz frame (0.2 s, the model drivers' slot clock) of a 512 x 288 render:
# wall / sim 0.75 unpaced vs 3.7 shipped (README "Speed", 2026-10-22). Each of the three env knobs overrides its default.
H=${HUMAN_H:-$([[ $HARM == 1 ]] && echo 288 || echo 1080)}; W=$(( (H * 16 / 9 + 7) / 8 * 8 )); CAMS=${HUMAN_CAMS:-front}; VIDEO=${HUMAN_VIDEO:-0}; NRE=${HUMAN_NRE:-default}; PHYS=${HUMAN_PHYS:-1}
HRES=${HUMAN_HARM_RES-$([[ $HARM == 1 ]] && echo "288 512")}
[[ $HARM == 1 && ! -f $NUREC/harmonizer/harmonizer_nontemporal.pt ]] && die "HUMAN_HARMONIZER=1 but $NUREC/harmonizer/harmonizer_nontemporal.pt is missing"
lever=()
if [[ $TRACK == pai ]]; then
  HOLD=1.7 SIM=20; FI=${HUMAN_FI_MS:-$([[ $HARM == 1 ]] && echo 200 || echo 100)}
  wiz=(+e2e_challenge=dev "scenes.scene_ids=[$quoted]" scenes.limit_to_first_n=0 "scenes.scene_cache=$NUREC"
       "scenes.scenes_csv=[$SRC/data/scenes/sim_scenes.csv,$SRC/data/scenes/sim_scenes_2604.csv]" "services.renderer.gpus=[$GPU]"
       "services.physics.gpus=[$GPU]" runtime.endpoints.physics.n_concurrent_rollouts=1)
  cam="{logical_id:camera_front_wide_120fov,height:$H,width:$W,frame_interval_us:$((FI * 1000)),shutter_duration_us:30000}"
  [[ $PHYS == 0 && $MODE == fast ]] && lever+=(runtime.simulation_config.physics_update_mode=NONE)
  mounts=(-v "$NUREC:$NUREC")
else
  HOLD=0.5 SIM=5.5; FI=${HUMAN_FI_MS:-500}
  cap=$(nvidia-smi -i "$GPU" --query-gpu=compute_cap --format=csv,noheader)
  wiz=(+e2e_challenge_nuplan=full runtime.nr_workers=1 "scenes.scene_ids=[$quoted]" scenes.limit_to_first_n=0 "services.renderer.gpus=[$GPU]")
  # gsplat's shipped arch list (8.9;9.0+PTX) has no kernel for an RTX 3090 (sm_86): the card's own arch, as docker/smoke.sh does
  [[ $(echo "$cap < 8.9" | bc) == 1 ]] && wiz+=("services.renderer.environments=[\"UV_LINK_MODE=copy\",\"TORCH_CUDA_ARCH_LIST=$cap\"]")
  cam="{logical_id:CAM_F0,height:$H,width:$W,frame_interval_us:$((FI * 1000))}"
  mounts=(-v "$NUPLAN:$NUPLAN:ro")
fi
if [[ $MODE == fast ]]; then
  [[ $CAMS == front ]] && lever+=("runtime.simulation_config.cameras=[$cam]")
  lever+=("eval.video.render_video=$([[ $VIDEO == 1 ]] && echo true || echo false)")
fi
wiz=(uv run alpasim_wizard "${wiz[@]}" runtime.endpoints.renderer.n_concurrent_rollouts=1 runtime.endpoints.driver.n_concurrent_rollouts=1
     runtime.endpoints.controller.n_concurrent_rollouts=1 defines.nre_cache_size=2 "wizard.baseport=$P0" wizard.run_method=NONE
     "wizard.log_dir=$R" "${lever[@]}" "$@")
[[ ${HUMAN_GT:-1} == 1 ]] && wiz+=(runtime.simulation_config.send_recording_ground_truth=true)
[[ $SERVE == 1 ]] && wiz+=(wizard.run_mode=SERVER "wizard.runtime_server_port=$RT")

phase driver "starting the driver container"
st "driver (human, phone port ${HUMAN_WHEEL_PORT:-47800}) in $IMG, 127.0.0.1:$DPORT, $n scenes, track $TRACK, mode $MODE, harmonizer $HARM"
docker rm -f "$N" >/dev/null 2>&1
docker run -d --name "$N" --init --network host --user "$(id -u):$(id -g)" -v "$REPO:$REPO:ro" -v "$OUT/driver:/logs" -v "$LIVE:$LIVE" \
  -e ALPASIM_DRIVER_HOST=127.0.0.1 -e "ALPASIM_DRIVER_PORT=$DPORT" -e ALPASIM_DRIVER_LOG_DIR=/logs -e "HUMAN_LIVE=$LIVE" -e "HUMAN_TRACK=$TRACK" \
  -e "HUMAN_N_SCENES=$([[ $SERVE == 1 ]] && echo 0 || echo "$n")" -e "HUMAN_SIM_S=$SIM" -e "HUMAN_HOLD_S=${HUMAN_HOLD_S:-$HOLD}" -e "HUMAN_START=${HUMAN_START:-throttle}" \
  -e "HUMAN_WAIT_MAX_S=${HUMAN_WAIT_MAX_S:-240}" -e "HUMAN_REALTIME=${HUMAN_REALTIME:-1.0}" -e "HUMAN_MODEL_REALTIME=${HUMAN_MODEL_REALTIME:-0}" -e "HUMAN_WHEEL_PORT=${HUMAN_WHEEL_PORT:-47800}" -e "HUMAN_RUN=$OUT" -e "HUMAN_CONTROL=${HUMAN_CONTROL:-}" \
  -e "HUMAN_HARMONIZER=$HARM" -e "HUMAN_TRACK_GATE=${HUMAN_TRACK_GATE:-1}" -e "HUMAN_FAST_ABORT=${HUMAN_FAST_ABORT:-$SERVE}" \
  -e "HUMAN_GIT=$(git -C "$REPO" rev-parse --short HEAD)" --entrypoint python "$IMG" "$REPO/experiments/alpasim_human/lib/human_driver.py" >/dev/null || die "docker run"
t0=$(date +%s)
until docker logs "$N" 2>&1 | grep -q "listening on"; do
  [[ $(docker inspect -f '{{.State.Running}}' "$N") == true ]] || die "driver exited before listening"; (( $(date +%s) - t0 > 60 )) && die "driver not listening"; sleep 0.5
done
( peak=0; while [[ ! -f $OUT/.stop ]]; do
    u=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits); (( u > peak )) && peak=$u
    echo "{\"t\": $(date +%s), \"card_mib\": $u, \"peak_mib\": $peak}" >> "$OUT/usage.jsonl"; sleep 2; done ) & sampler=$!

phase wizard "wizard: writing the configs for $n scenes"
st "wizard in $BASE (run_method NONE), render card $GPU, baseport $P0"
printf '%q ' "${wiz[@]}" > "$OUT/wizard_cmd.txt"; echo >> "$OUT/wizard_cmd.txt"
docker run --rm --gpus all -v "$SRC:$SRC" -v "$R:$R" "${mounts[@]}" -e "PYTHONPATH=$SRC/src/wizard:$SRC/src/utils" -e HF_HUB_OFFLINE=1 \
  -e ALPASIM_DRIVER_HOST=127.0.0.1 -e "ALPASIM_DRIVER_PORT=$DPORT" -e "ALPASIM_NUPLAN_ROOT=$NUPLAN" "$BASE" \
  bash -c 'umask 0000; git config --global --add safe.directory "*"; cd /repo && "$@"' wizard "${wiz[@]}" > "$OUT/wizard.log" 2>&1 || die "wizard failed, see wizard.log"
[[ -f $R/docker-compose.yaml ]] || die "no docker-compose.yaml"
# PAI: the public leaderboard's renderer is unharmonised (ec2 preset), as pai_run.sh, unless HUMAN_HARMONIZER=1 (flag kept, weights
# mounted, HUMAN_HARM_RES its resolution); HUMAN_NRE=nrend swaps the NRE backend (fast mode)
# HUMAN_JIT_CACHE=1 (default): the renderer's JIT caches live on the host, so kernels compiled once are reused by the next launch
# (cache only, outputs unchanged). Measured: each artifact revision's first PAI scene spent ~40 s in the CUDA driver's PTX JIT
# (/home/.nv/ComputeCache in the NRE container), lost with the container.
JIT=${HUMAN_JIT_DIR:-$D/cache/alpasim_human/jit}; [[ ${HUMAN_JIT_CACHE:-1} == 1 ]] && mkdir -p "$JIT" && chmod 777 "$JIT"
python3 - "$R/docker-compose.yaml" "$([[ $MODE == fast ]] && echo "$NRE" || echo default)" "$([[ ${HUMAN_JIT_CACHE:-1} == 1 ]] && echo "$JIT")" \
  "$HARM" "$NUREC/harmonizer" "$HRES" <<'PY' || die "compose patch"
import sys, yaml
f, nre, jit, harm, cache, hres = sys.argv[1:]
d = yaml.safe_load(open(f))
for k, s in d["services"].items():
    if not k.startswith("renderer"):
        continue
    if "pycena" in " ".join(s["command"]):
        if harm == "1":
            assert any("--enable-harmonizer" in c for c in s["command"]), "the preset has no --enable-harmonizer"
            s.setdefault("volumes", []).append(f"{cache}:/home/.cache/nre/harmonizer")
        s["command"] = [(c if harm == "1" else c.replace(" --enable-harmonizer", "")) +
                        (f" --renderer={nre}" if nre != "default" and "serve-grpc" in c else "") +
                        (f" --harmonizer-resolution {hres}" if harm == "1" and hres and "serve-grpc" in c else "") for c in s["command"]]
    if jit:
        s.setdefault("volumes", []).append(f"{jit}:/jitcache")
        s["environment"] = (s.get("environment") or []) + ["CUDA_CACHE_PATH=/jitcache/nv", "CUDA_CACHE_MAXSIZE=4294967296",
                                                           "TORCH_EXTENSIONS_DIR=/jitcache/torch_extensions", "TORCHINDUCTOR_CACHE_DIR=/jitcache/inductor"]
yaml.safe_dump(d, open(f, "w"), sort_keys=False)
PY
grep -A1 "device_ids:" "$R/docker-compose.yaml" | grep -E "^ *- '[0-9]+'" | grep -vq "'$GPU'" && die "compose puts a service on a card other than $GPU"

if [[ $SERVE == 1 ]]; then                                         # resident: services up, scenes come one request at a time
  dc() { docker compose -p "$NAME" -f "$R/docker-compose.yaml" "$@"; }
  phase containers "starting the containers"; st "docker compose up -d ($NAME, resident; runtime daemon on $RT)"
  dc up -d > "$OUT/compose.log" 2>&1 || die "compose up failed, see compose.log"
  dc logs -f --no-color >> "$OUT/compose.log" 2>&1 & logger=$!
  # service:port of every container (the gRPC port in its command line)
  svc=$(python3 -c 'import re,sys,yaml
for k, s in yaml.safe_load(open(sys.argv[1]))["services"].items():
    m = re.search(r"(?:--port[= ]|listen-address=\S*:)(\d+)", " ".join(s.get("command") or []))
    m and print("%s:%s" % (k.rsplit("-", 1)[0], m.group(1)))' "$R/docker-compose.yaml")
  t1=$(date +%s)
  while :; do
    wait_on=(); for sp in $svc; do ss -ltn | grep -q ":${sp##*:} " || wait_on+=("${sp%%:*}"); done
    dead=$(dc ps -a --status exited --format '{{.Service}}' | paste -sd' '); [[ -n $dead ]] && die "$dead exited during the start, see compose.log"
    (( ${#wait_on[@]} )) || break
    w=" ${wait_on[*]} "; why=""
    [[ $w == *" renderer "* ]] && why=" (the renderer loads its model; the first start also compiles kernels)"
    [[ $w != *" renderer "* && $w == *" runtime "* ]] && why=" (the runtime connects to the services)"
    phase services "containers up; waiting for ${wait_on[*]}$why"
    (( $(date +%s) - t1 > 900 )) && die "services not up after 900 s: ${wait_on[*]}"
    sleep 1
  done
  st "services up after $(( $(date +%s) - t1 )) s; feeder on 127.0.0.1:$RT"
  docker rm -f "$NAME-feed" >/dev/null 2>&1
  docker run -d --name "$NAME-feed" --init --network host --user "$(id -u):$(id -g)" -v "$OUT:$OUT" -v "$REPO:$REPO:ro" -v "$LIVE:$LIVE" \
    -e "HUMAN_IDLE_S=${HUMAN_IDLE_S:-1200}" --entrypoint python "$IMG" "$here/human_feed.py" "$OUT" "127.0.0.1:$RT" "$LIVE" >/dev/null || die "feeder docker run"
  docker logs -f "$NAME-feed" > "$OUT/feed.log" 2>&1 &
  phase ready "ready: renderer, controller, runtime and driver up"
  launcher running "resident $TRACK stack ready$([[ $HARM == 1 ]] && echo ", harmonizer on")" "$OUT"
  while [[ $(docker inspect -f '{{.State.Running}}' "$NAME-feed" 2>/dev/null) == true ]]; do
    dead=$(dc ps -a --status exited --format '{{.Service}}' | paste -sd' '); [[ -n $dead ]] && die "$dead exited, see compose.log"
    [[ $(docker inspect -f '{{.State.Running}}' "$N" 2>/dev/null) == true ]] || die "the driver exited, see driver.log"
    sleep 2
  done
  frc=$(docker inspect -f '{{.State.ExitCode}}' "$NAME-feed" 2>/dev/null || echo 1)
  kill "$logger" 2>/dev/null; trap - INT TERM
  (( frc == 0 )) || die "the feeder exited with $frc, see feed.log"
  phase stopped "stopped (idle or asked to stop)"; stop_all; wait "$sampler"; rm -f "$OUT/.stop"
  mo=$(docker inspect -f '{{index .Config.Labels "human.out"}}' "$PFX-model" 2>/dev/null)   # model mode's driver, kept warm by the loop
  [[ -n $mo ]] && { docker logs "$PFX-model" > "$mo/driver.log" 2>&1; docker rm -f "$PFX-model" >/dev/null 2>&1; }
  date > "$OUT/DONE"; st "resident stack stopped"; exit 0
fi

st "docker compose up ($NAME): drive the scenes; tmux human:alpa-viewer shows them"
launcher running "$n scenes$([[ $HARM == 1 ]] && echo ", harmonizer on")" "$OUT"
t1=$(date +%s)
(cd "$R" && docker compose -p "$NAME" -f docker-compose.yaml up --exit-code-from runtime-0) > "$OUT/compose.log" 2>&1; rc=$?
wall=$(( $(date +%s) - t1 ))
stop_all; wait "$sampler"; rm -f "$OUT/.stop"; trap - INT TERM
(( rc == 0 )) || die "compose exited with $rc after $wall s, see compose.log"
[[ -f $R/aggregate/results-summary.json ]] || die "no results-summary.json"
python3 -c 'import json,sys; k=sys.argv[1::2]; v=sys.argv[2::2]; print(json.dumps(dict(zip(k, v))))' track "$TRACK" mode "$MODE" scenes "$n" wall_s "$wall" \
  gpu "$GPU" cams "$CAMS" h "$H" w "$W" fi_ms "$FI" harm_res "$HRES" video "$VIDEO" nre "$NRE" phys "$PHYS" harmonizer "$HARM" gt "${HUMAN_GT:-1}" image "$IMG" base "$BASE" \
  peak_mib "$(tail -1 "$OUT/usage.jsonl" | python3 -c 'import json,sys; print(json.load(sys.stdin)["peak_mib"])')" > "$OUT/run.json"
python3 "$here/human_results.py" "$OUT" || st "results table failed"
date > "$OUT/DONE"; st "done: $n scenes in $wall s"; launcher finished "$n scenes done: $OUT/results.md" "$OUT"
