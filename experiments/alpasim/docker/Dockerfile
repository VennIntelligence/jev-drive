# Submission image of the AlpaSim nuPlan track: one image, every openpilot + adapter driver of experiments/alpasim/lib, the driver family
# and checkpoint picked at run time (serve.py: JEV_TAG / JEV_DRIVER). Built by build.sh from a staged context; never by hand.
# Layers, most stable first, so a new checkpoint or default costs one thin layer (no re-pull, no re-push of the rest):
#   python + torch -> pinned packages -> alpasim_grpc -> openpilot base model -> driver code -> stable checkpoints -> extra checkpoints -> defaults
# Conventions follow the shipped samples (e2e_challenge/sample_submission_simscale_navsim_transfuser/Dockerfile at 0bb4c4b): gRPC on
# $ALPASIM_DRIVER_HOST:$ALPASIM_DRIVER_PORT (0.0.0.0:6789), uid 10001, no HEALTHCHECK, every cache and log under /tmp, SIGTERM stops it.
# Packages are the versions of the box's serving env (envs/op-train, where every native closed-loop number was measured); torch is the
# same release built for CUDA 12.6 instead of 13.0 so that an older host driver still loads it. requirements.lock (uv, with hashes) is
# what gets installed: from the named build context `wheels` when build.sh is given WHEELS=<dir> (the locked files fetched by
# wheels.py; uv then runs offline), otherwise from PyPI and the PyTorch index. Either way the hashes decide the content.
ARG PY=3.11
FROM python:${PY}-slim-bookworm AS grpc
# The gRPC package as the samples install it (its build hook compiles the protos), built here so its build tools stay out of the image.
COPY alpasim/src/grpc /src/grpc
RUN --mount=type=cache,target=/root/.cache/pip PIP_NO_CACHE_DIR=0 pip wheel --no-deps -w /whl /src/grpc

FROM python:${PY}-slim-bookworm
ARG CUDA=cu126
ARG UV_SRC=""
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HOME=/app TMPDIR=/tmp XDG_CACHE_HOME=/tmp/.cache TORCH_HOME=/tmp/torch TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    HF_HOME=/tmp/huggingface HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/matplotlib CUDA_CACHE_PATH=/tmp/nv NUMBA_CACHE_DIR=/tmp/numba \
    DATA_DIR=/app/data ALPASIM_SRC=/app/alpasim ALPASIM_DRIVER_LOG_DIR=/tmp/alpasim-driver \
    ALPASIM_DRIVER_HOST=0.0.0.0 ALPASIM_DRIVER_PORT=6789 ALPASIM_DRIVER_GRPC_WORKERS=8 \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
WORKDIR /app
COPY requirements.lock /app/requirements.lock
RUN --mount=type=bind,from=ghcr.io/astral-sh/uv:latest,source=/uv,target=/bin/uv \
    --mount=type=bind,from=wheels,target=/wheels --mount=type=cache,target=/root/.cache/uv \
    UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy uv pip install --system --require-hashes ${UV_SRC:---extra-index-url https://download.pytorch.org/whl/${CUDA} --index-strategy unsafe-best-match} \
    -r /app/requirements.lock
COPY --from=grpc /whl /tmp/whl
# --no-deps: the package pins protobuf < 5 for its build tools; the generated stubs run on the serving env's protobuf (as on the box).
RUN --mount=type=bind,from=ghcr.io/astral-sh/uv:latest,source=/uv,target=/bin/uv \
    uv pip install --system --no-cache --no-deps /tmp/whl/*.whl && rm -rf /tmp/whl \
    && python -c "import alpasim_grpc, torch; assert alpasim_grpc.__version__ == (0, 54, 0), alpasim_grpc.__version__; print(torch.__version__)"

# openpilot Cinque (the frozen base of every driver); jevdrive.op_adapt reads it from ~/data/models/openpilot, HOME=/app.
COPY models /app/data/models
# Driver code: the import closure of the drivers (closure.txt) at its repo-relative paths, the two helper modules of the shipped LTF
# sample the drivers import unchanged, and this directory's entry point and probe.
COPY alpasim/e2e_challenge /app/alpasim/e2e_challenge
COPY alpasim/LICENSE /app/alpasim/LICENSE
COPY jev-drive /app/jev-drive
COPY serve.py probe.py verify.py /app/
RUN groupadd --gid 10001 alpasim \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /app --shell /usr/sbin/nologin alpasim

# Checkpoints: the stable set, then the tags added since (build.sh TAGS / EXTRA). verify.py imports every driver, reads each checkpoint
# and writes tags.json (tag -> driver family) for serve.py. build.sh stages the context world-readable (the image runs as uid 10001).
COPY ckpt /app/data/runs/op_parity/runs
COPY ckpt-extra /app/data/runs/op_parity/runs
RUN python /app/verify.py

# Defaults of this build: the one-line switch of what a bare `docker run` (the organisers' run) serves.
ARG TAG=P2H10-F-s0
ARG DRIVER=auto
ARG GIT_HASH=local
# The served speed profile (lib/serve_fix.py): VCONT = seconds of the speed-continuous blend (0 = off), LEAD = 1 for the openpilot lead limit.
ARG VCONT=0
ARG LEAD=0
ENV JEV_TAG=${TAG} JEV_DRIVER=${DRIVER} SH30_GIT_HASH=${GIT_HASH} JEV_VCONT=${VCONT} JEV_LEAD=${LEAD}
EXPOSE 6789
STOPSIGNAL SIGTERM
USER 10001:10001
CMD ["python", "/app/serve.py"]
