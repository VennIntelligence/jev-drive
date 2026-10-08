# Sourced by the alpasim scripts: toolchain and uv settings for the native (Docker-free) AlpaSim env on the box.
export RUSTUP_HOME=$DATA_DIR/tools/rust/rustup CARGO_HOME=$DATA_DIR/tools/rust/cargo
export PATH=$CARGO_HOME/bin:/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8 CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11
export UV_LINK_MODE=copy UV_PYTHON=3.12
# Workspace members the nuPlan/MTGS track starts (wizard, runtime + eval, controller, MTGS renderer). The shipped
# image also installs the in-repo drivers (vam, alpamayo) and physics, which this track never runs.
export ALPASIM_MEMBERS="src/grpc src/utils src/utils_rs src/runtime src/eval src/wizard src/controller src/plugins src/trajdata plugins/mtgs[server]"
# gsplat compiles its CUDA kernels on first use (Dockerfile: gcc-11, ninja, MAX_JOBS=4); keep the build off the system disk.
export MAX_JOBS=4 TORCH_EXTENSIONS_DIR=$DATA_DIR/cache/torch_extensions/alpasim
