#!/usr/bin/env bash
# WOD-E2E data path: index the slim shards, check them, report stats and ego-only baselines (docs/waymo-e2e.md).
# Usage (on the box, in tmux): scripts/tmux_run.sh waymo-prep scripts/waymo_prepare.sh [command ...]
#   scripts/waymo_prepare.sh                      # index -> check -> report (default)
#   scripts/waymo_prepare.sh index --workers 8    # rescan only shards that have no cache
#   scripts/waymo_prepare.sh features --cams front3 --batch-size 4
#   scripts/waymo_prepare.sh bench --limit 96     # ms/frame, VRAM, tokens per input choice
# Incremental: rerun it whenever new shards have landed. Options: python -m jevdrive.waymo --help.
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set, see docs/storage.md}"

WOD_COMMIT=99a4cb3ff07e2fe06c2ce73da001f850f628e45a  # keep in sync with scripts/download_waymo_e2e.sh
ENV=$DATA_DIR/envs/waymo
export WAYMO_PROTO_GEN=$ENV/gen
export UV_PROJECT_ENVIRONMENT="$DATA_DIR/envs/jevdrive" PYTHONWARNINGS=ignore::FutureWarning
cd "$(dirname "$0")/.."

# The download script compiles the data proto; the submission proto is only needed here. Same venv, same gen dir.
if [[ ! -f $WAYMO_PROTO_GEN/waymo_open_dataset/protos/end_to_end_driving_submission_pb2.py ]]; then
  [[ -x $ENV/bin/python ]] || { echo "run scripts/download_waymo_e2e.sh once first (it builds $ENV)" >&2; exit 1; }
  mkdir -p "$WAYMO_PROTO_GEN" "$DATA_DIR/tmp"
  src=$(mktemp -d -p "$DATA_DIR/tmp")
  (set +u; source /etc/network_turbo >/dev/null
   curl -fsSL "https://codeload.github.com/waymo-research/waymo-open-dataset/tar.gz/$WOD_COMMIT" \
     | tar xz -C "$src" --strip-components=2 --wildcards '*/src/waymo_open_dataset/*.proto')
  (cd "$src" && "$ENV/bin/python" -m grpc_tools.protoc -I. --python_out="$WAYMO_PROTO_GEN" \
     waymo_open_dataset/protos/end_to_end_driving_{data,submission}.proto waymo_open_dataset/dataset.proto \
     waymo_open_dataset/label.proto waymo_open_dataset/protos/{map,vector,keypoint}.proto)
  rm -rf "$src"
fi

uv sync -q
if (( $# )); then
  exec uv run python -u -m jevdrive.waymo "$@"
fi
for cmd in index check report; do
  echo "==> $cmd"
  uv run python -u -m jevdrive.waymo "$cmd"
done
