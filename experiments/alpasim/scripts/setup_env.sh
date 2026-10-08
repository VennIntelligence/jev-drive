#!/usr/bin/env bash
# Build the native AlpaSim env on the box (no Docker there): clone at the challenge-deployed commit, user-level
# Rust toolchain, compiled protos, uv env with the services the nuPlan/MTGS track starts. Their source is untouched.
# Usage (box): scripts/tmux_run.sh alpasim-env experiments/alpasim/scripts/setup_env.sh
set -euo pipefail
SRC=${ALPASIM_SRC:-$DATA_DIR/third_party/alpasim}
COMMIT=${ALPASIM_COMMIT:-0bb4c4bfe10951ea5589aa8ea514e422cf3c3506}
OUT=$DATA_DIR/runs/alpasim/setup; mkdir -p "$OUT"; rm -f "$OUT/DONE" "$OUT/ERROR"
exec > >(tee -a "$OUT/log.txt") 2>&1
trap 'echo "failed at line $LINENO" > "$OUT/ERROR"' ERR
source "$(dirname "$0")/env.sh"
# GitHub (clone, uv git / archive sources) goes through Clash, PyPI direct from the Aliyun mirror (docs/network-proxy.md).
clash-start >/dev/null
proxy() { env http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890 \
  no_proxy=localhost,127.0.0.1,.aliyun.com,.aliyuncs.com,.rsproxy.cn "$@"; }

echo clone > "$OUT/STATUS"
[[ -d $SRC/.git ]] || proxy git clone --branch e2e_challenge https://github.com/NVlabs/alpasim "$SRC"
git -C "$SRC" checkout -q "$COMMIT"

echo rust > "$OUT/STATUS"
if ! command -v cargo >/dev/null; then
  export RUSTUP_DIST_SERVER=https://rsproxy.cn RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup
  curl -sSf --retry 3 https://rsproxy.cn/rustup-init.sh | sh -s -- -y --no-modify-path --profile minimal
  printf '[source.crates-io]\nreplace-with = "rsproxy-sparse"\n[source.rsproxy-sparse]\nregistry = "sparse+https://rsproxy.cn/index/"\n' \
    > "$CARGO_HOME/config.toml"
fi

# Their route is `uv sync --extra all --extra mtgs` (Dockerfile). That resolves the whole workspace, including the
# in-repo drivers' GitHub sources (vam, alpamayo, a lightning archive) the box cannot fetch reliably, and the root
# pyproject's `exclude-newer` needs upload dates the Aliyun mirror does not serve. So the members the nuPlan/MTGS
# track starts are installed editable into the same .venv, without the root [tool.uv] settings (--no-config); the
# repo has no uv.lock, so versions float either way. torch < 2.10 keeps the cu128 build that matches the box's
# nvcc 12.8, which gsplat's first-use kernel build needs.
echo sync > "$OUT/STATUS"
cd "$SRC"
[[ -x .venv/bin/python ]] || uv venv --python 3.12 .venv
printf 'setuptools<82\ntorch<2.10\n' > "$OUT/constraints.txt"   # the first is the root pyproject's constraint
uv pip install --no-config --python .venv/bin/python --default-index https://mirrors.aliyun.com/pypi/simple \
  -c "$OUT/constraints.txt" $(printf -- '-e %s ' $ALPASIM_MEMBERS)

echo protos > "$OUT/STATUS"
(cd src/grpc && UV_NO_SYNC=1 uv run compile-protos)
uv pip freeze --python .venv/bin/python > "$OUT/freeze.txt"
"$SRC/.venv/bin/python" - <<'PY' | tee "$OUT/versions.txt"
import torch, gsplat, numpy, grpc
print("torch", torch.__version__, "cuda", torch.version.cuda, "archs", torch.cuda.get_arch_list())
print("gsplat", gsplat.__version__, "numpy", numpy.__version__, "grpcio", grpc.__version__)
PY
du -sh "$SRC/.venv" | tee -a "$OUT/versions.txt"
date > "$OUT/DONE"; echo done > "$OUT/STATUS"
