# Sourced by the alpasim scripts: toolchain and uv settings for the native (Docker-free) AlpaSim env on the box.
export RUSTUP_HOME=$DATA_DIR/tools/rust/rustup CARGO_HOME=$DATA_DIR/tools/rust/cargo
export PATH=$CARGO_HOME/bin:/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8 CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11
export UV_LINK_MODE=copy UV_PYTHON=3.12
# Services of the nuPlan/MTGS track only. The shipped image syncs `--extra all --extra mtgs`; `all` adds the
# in-repo drivers (vam, alpamayo) and physics, which this track never starts.
export ALPASIM_EXTRAS="--extra wizard --extra runtime --extra controller --extra eval --extra grpc --extra utils --extra plugins --extra mtgs"
export UVSYNC="uv sync --default-index https://mirrors.aliyun.com/pypi/simple"
# gsplat compiles its CUDA kernels on first use (Dockerfile: gcc-11, ninja, MAX_JOBS=4); keep the build off the system disk.
export MAX_JOBS=4 TORCH_EXTENSIONS_DIR=$DATA_DIR/cache/torch_extensions/alpasim
