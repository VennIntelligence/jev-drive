#!/usr/bin/env bash
# Download WOD-E2E v1.0.0 and keep only the three front cameras (docs/waymo-e2e.md).
# Usage (on the box, in tmux): scripts/tmux_run.sh waymo scripts/download_waymo_e2e.sh [split ...] [--option ...]
#   splits: small val train test (default: all, in that order); options: see scripts/waymo_e2e.py download -h
#   scripts/download_waymo_e2e.sh inspect <tfrecord>    # frame/camera/trajectory facts of one shard
# Idempotent and resumable: finished shards (manifest.csv) are skipped; rerun after the disk guard stops it.
# Sets up its own venv ($DATA_DIR/envs/waymo: protobuf + google-crc32c, no TensorFlow) and compiled protos.
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set, see docs/storage.md}"

WOD_COMMIT=99a4cb3ff07e2fe06c2ce73da001f850f628e45a  # waymo-research/waymo-open-dataset, protos only
ENV=$DATA_DIR/envs/waymo
export WAYMO_PROTO_GEN=$ENV/gen
export CLOUDSDK_PYTHON=$DATA_DIR/tools/google-cloud-sdk/platform/bundledpythonunix/bin/python3
export PATH=$DATA_DIR/tools/google-cloud-sdk/bin:$PATH
repo=$(cd "$(dirname "$0")/.." && pwd)

if [[ ! -x $ENV/bin/python ]]; then  # pip mirrors are domestic: install without any proxy
  uv venv -q "$ENV" --python 3.12
  VIRTUAL_ENV=$ENV uv pip install -q protobuf grpcio-tools google-crc32c tqdm
fi
if [[ ! -f $WAYMO_PROTO_GEN/waymo_open_dataset/protos/end_to_end_driving_data_pb2.py ]]; then
  mkdir -p "$WAYMO_PROTO_GEN" "$DATA_DIR/tmp"
  src=$(mktemp -d -p "$DATA_DIR/tmp")
  (set +u; source /etc/network_turbo >/dev/null
   curl -fsSL "https://codeload.github.com/waymo-research/waymo-open-dataset/tar.gz/$WOD_COMMIT" \
     | tar xz -C "$src" --strip-components=2 --wildcards '*/src/waymo_open_dataset/*.proto')
  (cd "$src" && "$ENV/bin/python" -m grpc_tools.protoc -I. --python_out="$WAYMO_PROTO_GEN" \
     waymo_open_dataset/protos/end_to_end_driving_data.proto waymo_open_dataset/dataset.proto \
     waymo_open_dataset/label.proto waymo_open_dataset/protos/{map,vector,keypoint}.proto)
  rm -rf "$src"
fi

if [[ ${1:-} == inspect ]]; then
  exec "$ENV/bin/python" "$repo/scripts/waymo_e2e.py" "$@"
fi

# Google OAuth only works through Clash: start it here, but keep it out of this process's env.
# The Python side passes the proxy to gcloud only, and uses it for data only with --route proxy.
set +u; source ~/.bashrc >/dev/null 2>&1; proxy_on >/dev/null; proxy_off >/dev/null 2>&1 || true; set -u
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
gcloud auth list --filter=status:ACTIVE --format='value(account)' | grep -q . \
  || { echo "gcloud is not logged in: run 'bash -l ~/data/tools/gauth.sh' in a tmux window" >&2; exit 1; }

exec "$ENV/bin/python" -u "$repo/scripts/waymo_e2e.py" download "$@"
