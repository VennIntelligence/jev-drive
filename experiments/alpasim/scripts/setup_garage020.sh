#!/usr/bin/env bash
# Env and checkpoint for the host's reference baseline py123d_garage v0.2.0 (ltf-base-joint64-68M, posttrain/model_0.pth, the recommended
# seed) as an AlpaSim PAI driver, natively instead of its submission image. Mirrors lib/alpasim/tools/submission.Dockerfile at v0.2.0 (c5db115):
# python 3.12 venv, `pyproject.toml --extra alpasim`, the garage package with --no-deps (non-editable), alpasim-grpc from our AlpaSim checkout
# (src/grpc is identical between their pin e121e59 and our 0bb4c4b). Their setup_alpasim.sh (clone + patch AlpaSim e121e59, install the
# garage into AlpaSim's venv) is NOT run: the live checkout stays untouched, and none of the three patches is needed by the driver
# (experiments/alpasim/results/garage_v020/README.md). Then a GPU-free startup check: the driver on CPU on a free port, get_version over gRPC.
# Usage (box, repo root): scripts/tmux_run.sh garage020-setup bash experiments/alpasim/scripts/setup_garage020.sh
# Downloads go direct (HF_BASE); if slow, rerun under `source /etc/network_turbo`, then `proxy_on` (docs/network-proxy.md); resumable.
# Needs the v0.2.0 source at $G (copied from a clone; GitHub is often unreachable from the box).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); source "$HERE/env.sh"
SRC=${ALPASIM_SRC:-$DATA_DIR/third_party/alpasim}
G=$DATA_DIR/third_party/py123d_garage-v0.2.0 TAG=v0.2.0 REV=56edbe90a71a3fd4baee2d5f2f2da0bb7e722364
ENV=$DATA_DIR/envs/garage020 CK=$DATA_DIR/checkpoints/py123d_garage/v0.2.0/ltf-base-joint64-68M
OUT=$DATA_DIR/runs/alpasim/garage_v020/setup; mkdir -p "$OUT" "$CK"; rm -f "$OUT/DONE" "$OUT/ERROR"
exec > >(tee -a "$OUT/log.txt") 2>&1
trap 'echo "failed at line $LINENO" > "$OUT/ERROR"' ERR
PIP="uv pip install --python $ENV/bin/python --default-index https://mirrors.aliyun.com/pypi/simple"
HF=${HF_BASE:-https://huggingface.co}/kesai-labs/py123d_garage_pretrained_checkpoints/resolve/$REV/v0.2.0/ltf-base-joint64-68M

[[ $(git -C "$G" describe --tags --exact-match) == "$TAG" && -z $(git -C "$G" status --porcelain) ]]
git -C "$G" log -1 --format='garage %h %ad %s' --date=short
[[ -x $ENV/bin/python ]] || uv venv --python 3.12 "$ENV"
$PIP -r "$G/pyproject.toml" --extra alpasim
$PIP --no-deps "$G"
$PIP --no-deps "$SRC/src/grpc"
get() {  # get <path under the checkpoint dir> <size in bytes>
  local f=$CK/$1; mkdir -p "$(dirname "$f")"
  [[ -s $f && $(stat -c %s "$f") == "$2" ]] || curl -fL --retry 8 --retry-all-errors -C - -o "$f" "$HF/$1"
  [[ $(stat -c %s "$f") == "$2" ]] || { echo "size of $1: $(stat -c %s "$f") != $2"; false; }; }
get README.md 3856; get posttrain/config.yaml 20238; get posttrain/sensor_rig_nurec.yaml 71579; get posttrain/model_0.pth 271220327
for s in 0 1 2; do get results/nurec/seed_$s/results-summary.json $((s == 0 ? 1062358 : s == 1 ? 1062239 : 1061613)); done
(cd "$CK" && sha256sum README.md posttrain/* results/nurec/*/*) | tee "$OUT/sha256.txt"
"$ENV/bin/python" -c "import torch, torchvision, timm, numpy, grpc, py123d, alpasim_grpc, importlib.metadata as m; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'tv', torchvision.__version__, 'timm', timm.__version__, 'numpy', numpy.__version__, \
'grpcio', grpc.__version__, 'py123d', m.version('py123d'), 'py123d-garage', m.version('py123d-garage'), 'archs', torch.cuda.get_arch_list())" | tee "$OUT/versions.txt"
uv pip freeze --python "$ENV/bin/python" > "$OUT/freeze.txt"

# Startup check on CPU: the shipped entry point through garage_driver.py (the close-record observer), then get_version.
T=$OUT/startup; rm -rf "$T"; mkdir -p "$T"
PORT=$("$ENV/bin/python" -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])")
(cd "$T" && CUDA_VISIBLE_DEVICES= HOME=$T TMPDIR=$T ALPASIM_DRIVER_LOG_DIR=$T/driver-logs HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 \
  exec "$ENV/bin/python" "$HERE/../../alpasim_proxy/scripts/garage_driver.py" hydra.run.dir=$T/alpasim_driver \
  "policy_config.evaluation_checkpoint_files=[$CK/posttrain/model_0.pth]" policy_config.evaluation_sensor_rig_file=$CK/posttrain/sensor_rig_nurec.yaml \
  git_hash=c5db115 device=cpu host=127.0.0.1 port=$PORT) > "$T/driver.log" 2>&1 &
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
