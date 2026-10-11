#!/usr/bin/env bash
# Env and checkpoint for the public py123d_garage AlpaSim driver (resnet34_v0.1.0 / model_0014.pth), natively instead of its submission
# image. Mirrors lib/alpasim/tools/submission.Dockerfile at v0.1.1: python 3.12 venv, `pyproject.toml --extra alpasim`, the garage package
# with --no-deps (non-editable), alpasim-grpc from the AlpaSim checkout (src/grpc is identical between their pin cd713e0 and our 0bb4c4b).
# Then a GPU-free startup check: the driver on CPU on a free port, get_version over gRPC (it waits for the policy load).
# Usage (box): scripts/tmux_run.sh garage-setup experiments/alpasim_proxy/scripts/setup_garage.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); source "$HERE/../../alpasim/scripts/env.sh"
SRC=${ALPASIM_SRC:-$DATA_DIR/third_party/alpasim}
G=$DATA_DIR/third_party/py123d_garage TAG=v0.1.1
ENV=$DATA_DIR/envs/garage CK=$DATA_DIR/models/garage/resnet34_v0.1.0
OUT=$DATA_DIR/runs/alpasim_proxy/setup_garage; mkdir -p "$OUT" "$CK"; rm -f "$OUT/DONE" "$OUT/ERROR"
exec > >(tee -a "$OUT/log.txt") 2>&1
trap 'echo "failed at line $LINENO" > "$OUT/ERROR"' ERR
PIP="uv pip install --python $ENV/bin/python --default-index https://mirrors.aliyun.com/pypi/simple"
HF=https://hf-mirror.com/kesai-labs/py123d_garage_pretrained_checkpoints/resolve/main/resnet34_v0.1.0

[[ -d $G/.git ]] || git clone https://github.com/kesai-labs/py123d_garage.git "$G"
git -C "$G" fetch --tags --quiet || true
git -C "$G" checkout --quiet --detach "$TAG"; git -C "$G" log -1 --format='garage %h %ad %s' --date=short
[[ -x $ENV/bin/python ]] || uv venv --python 3.12 "$ENV"
$PIP -r "$G/pyproject.toml" --extra alpasim
$PIP --no-deps "$G"
$PIP --no-deps "$SRC/src/grpc"
for f in config.yaml sensor_rig_0.yaml model_0014.pth; do
  [[ -s $CK/$f && ! ( $f == model_0014.pth && $(stat -c %s "$CK/$f") != 247950835 ) ]] || curl -fL --retry 5 -C - -o "$CK/$f" "$HF/$f"
done
[[ $(stat -c %s "$CK/model_0014.pth") == 247950835 ]]
(cd "$CK" && sha256sum model_0014.pth config.yaml sensor_rig_0.yaml) | tee "$OUT/sha256.txt"
"$ENV/bin/python" -c "import torch, torchvision, timm, numpy, grpc, py123d, alpasim_grpc, importlib.metadata as m; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'tv', torchvision.__version__, 'timm', timm.__version__, 'numpy', numpy.__version__, \
'grpcio', grpc.__version__, 'py123d', m.version('py123d'), 'py123d-garage', m.version('py123d-garage'), 'archs', torch.cuda.get_arch_list())" | tee "$OUT/versions.txt"
uv pip freeze --python "$ENV/bin/python" > "$OUT/freeze.txt"

# Startup check on CPU: the shipped entry point through garage_driver.py, then get_version.
T=$OUT/startup; rm -rf "$T"; mkdir -p "$T"
PORT=$("$ENV/bin/python" -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])")
(cd "$T" && CUDA_VISIBLE_DEVICES= HOME=$T TMPDIR=$T ALPASIM_DRIVER_LOG_DIR=$T/driver-logs HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 \
  exec "$ENV/bin/python" "$HERE/garage_driver.py" hydra.run.dir=$T/alpasim_driver \
  policy_config.evaluation_checkpoint_file=$CK/model_0014.pth policy_config.evaluation_sensor_rig_file=$CK/sensor_rig_0.yaml \
  git_hash=288d34c device=cpu host=127.0.0.1 port=$PORT) > "$T/driver.log" 2>&1 &
DP=$!
"$ENV/bin/python" - "$PORT" <<'PY'
import sys, time, grpc
from alpasim_grpc.v0 import common_pb2, egodriver_pb2_grpc
ch = grpc.insecure_channel(f"127.0.0.1:{sys.argv[1]}")
grpc.channel_ready_future(ch).result(timeout=300)
t = time.time()
v = egodriver_pb2_grpc.EgodriverServiceStub(ch).get_version(common_pb2.Empty(), timeout=600)
print(f"get_version after {time.time() - t:.1f} s:", v.version_id, v.git_hash, v.grpc_api_version)
PY
kill $DP; wait $DP || true
tail -5 "$T/driver.log"
du -sh "$ENV" "$CK" | tee -a "$OUT/versions.txt"
date > "$OUT/DONE"
