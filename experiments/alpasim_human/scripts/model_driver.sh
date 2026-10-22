#!/usr/bin/env bash
# Model driver service for the human exam's model mode (README "Model mode"): one checkpoint served by the driver of a submission image,
# beside the resident stack, reached by lib/human_driver.py at 127.0.0.1:HUMAN_MODEL_PORT. Called by scripts/human_loop.py.
#   model_driver.sh pai|nuplan <tag> <out dir>        -> exits 0 once the driver listens (container $PFX-model keeps running), 1 on failure
# Image = the driver code and the served configuration (its baked ENV): PAI the submitted p1021a image (NTV0-L18-s0 + ACC + GAP 0.7,2.7,
# JEV_TRACK gate by default), nuPlan the submitted native JTN-L18-s1 image (every serving switch off). The tag's checkpoint directory
# /data/runs/op_parity/runs/<tag> is mounted where the image keeps its own (SH30_TAG = tag); the image's serve.py is bypassed only to
# wrap the same driver in lib/model_view.py (the model's own input frames for the viewer; display only).
# Container restrictions as the official runner's local copy (read-only root, /tmp 2 GB, no new privileges), host network (loopback only).
# Env: HUMAN_GPU (0), HUMAN_MODEL_PORT (6791), HUMAN_LIVE, HUMAN_PREFIX (hum), HUMAN_MODEL_IMG_PAI / HUMAN_MODEL_IMG_NUPLAN,
#   HUMAN_MODEL_ENV (extra `-e K=V`, space separated, e.g. "-e JEV_ACC=0"), DATA_DIR (/data).
# Labels human.{track,tag,port,gpu,out} let a later loop reuse a running container (scripts/human_loop.py ModelSvc).
# Writes <out>/{driver.log (docker logs at stop), logs/ (the driver's drive.jsonl / start.jsonl), info.json}.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$here/../../.." && pwd)
TRACK=$1 TAG=$2 OUT=$(mkdir -p "$3" && cd "$3" && pwd)
D=${DATA_DIR:-/data}; PFX=${HUMAN_PREFIX:-hum}; N=$PFX-model; PORT=${HUMAN_MODEL_PORT:-6791}
LIVE=${HUMAN_LIVE:-$D/runs/alpasim_human/live}; GPU=${HUMAN_GPU:-0}; CK=$D/runs/op_parity/runs/$TAG
case $TRACK in
  pai) IMG=${HUMAN_MODEL_IMG_PAI:-jev-alpasim:ntv0-l18-s0-0d417ee4-pai-vc1.0-lead-acc-gap0.7_2.7} ;;
  nuplan) IMG=${HUMAN_MODEL_IMG_NUPLAN:-jev-alpasim:jtn-l18-s1-ba3ef44d} ;;
  *) echo "track pai | nuplan"; exit 1 ;;
esac
[[ -f $CK/ckpt-final.pt ]] || { echo "no checkpoint $CK/ckpt-final.pt"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "image $IMG missing"; exit 1; }
docker rm -f "$N" >/dev/null 2>&1
ss -ltn | grep -q ":$PORT " && { echo "port $PORT is taken"; exit 1; }
mkdir -p "$OUT/logs" "$LIVE"; chmod 777 "$OUT/logs"
# shellcheck disable=SC2086
docker run -d --name "$N" --init --cap-drop ALL --security-opt no-new-privileges:true --read-only --pids-limit 1024 --memory 32g --cpus 8 \
  --tmpfs /tmp:rw,nosuid,nodev,size=2g,mode=1777 --tmpfs /run:rw,nosuid,nodev,size=64m,mode=0755 --network host --gpus "device=$GPU" \
  --user "$(id -u):$(id -g)" -v "$REPO:$REPO:ro" -v "$CK:/app/data/runs/op_parity/runs/$TAG:ro" -v "$OUT/logs:/logs" -v "$LIVE:$LIVE" \
  --label "human.track=$TRACK" --label "human.tag=$TAG" --label "human.port=$PORT" --label "human.gpu=$GPU" --label "human.out=$OUT" \
  -e ALPASIM_DRIVER_HOST=127.0.0.1 -e "ALPASIM_DRIVER_PORT=$PORT" -e ALPASIM_DRIVER_LOG_DIR=/logs -e "SH30_TAG=$TAG" -e "HUMAN_LIVE=$LIVE" \
  ${HUMAN_MODEL_ENV:-} --entrypoint python "$IMG" "$REPO/experiments/alpasim_human/lib/model_view.py" "$TRACK" >/dev/null || { echo "docker run failed"; exit 1; }
t0=$(date +%s)
until docker logs "$N" 2>&1 | grep -q "listening on"; do
  if [[ $(docker inspect -f '{{.State.Running}}' "$N" 2>/dev/null) != true ]] || (( $(date +%s) - t0 > 600 )); then
    docker logs "$N" > "$OUT/driver.log" 2>&1; docker rm -f "$N" >/dev/null 2>&1
    echo "the model driver did not start: $(grep -m1 -iE 'error|exception' "$OUT/driver.log" || tail -1 "$OUT/driver.log")"; exit 1
  fi
  sleep 0.5
done
env=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$IMG" | grep -E '^(JEV_|SH30_COMPILE|PAI_)' | paste -sd' ')
python3 -c 'import json,sys; print(json.dumps(dict(zip(sys.argv[1::2], sys.argv[2::2]))))' track "$TRACK" tag "$TAG" image "$IMG" \
  image_id "$(docker image inspect -f '{{.Id}}' "$IMG")" served_env "$env ${HUMAN_MODEL_ENV:-}" port "$PORT" gpu "$GPU" \
  ready_s "$(( $(date +%s) - t0 ))" container "$N" > "$OUT/info.json"
echo "model driver $TAG ($TRACK) listening on 127.0.0.1:$PORT after $(( $(date +%s) - t0 )) s"
