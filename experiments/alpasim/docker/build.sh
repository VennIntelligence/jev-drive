#!/usr/bin/env bash
# Build the AlpaSim submission image (Dockerfile here) on a Docker host that holds the repo, the AlpaSim checkout and the weights at the
# box's relative paths under $DATA_DIR (Tokyo: /data). Stages a minimal context (closure.txt, src/grpc, two LTF-sample helpers,
# cinque.ort.onnx, one ckpt-final.pt per tag), builds, prints size and layers. Nothing else of the repo enters the image.
#   experiments/alpasim/docker/build.sh                               # default: serves P2H10-F-s0
#   TAG=AP2-AB-s0 experiments/alpasim/docker/build.sh                 # the one-line switch: same layers, another default
#   EXTRA="OT10a05-F-s0" TAG=OT10a05-F-s0 .../build.sh                # a new checkpoint: one thin layer on top of the cached ones
# Env: TAGS (stable checkpoint layer), EXTRA (thin layer), TAG (served by default; several joined by + = ensemble), DRIVER (auto | sh30 |
# ap2 | ens), VCONT / LEAD (the serving switches baked in as JEV_VCONT / JEV_LEAD, lib/serve_fix.py; default 0 / 0 = off),
# IMAGE (repository, default jev-alpasim), VERSION (image tag, default <TAG>-<git short hash>[-vc<VCONT>][-lead]),
# WHEELS (dir of pre-fetched wheels, see wheels.py: the build then installs offline from it; default $DATA_DIR/cache/jev_wheels when it
# exists, WHEELS= forces the indexes, whose triton wheel does not match requirements.lock as of 2026-10-09), PROXY (http proxy for the build's
# downloads, e.g. http://127.0.0.1:7890; uses the host network), DATA_DIR, ALPASIM_SRC.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); repo=$(cd "$here/../../.." && pwd)
: "${DATA_DIR:?DATA_DIR}"
SRC=${ALPASIM_SRC:-$DATA_DIR/third_party/alpasim}
TAGS=${TAGS:-"P2H10-F-s0 AP2-AB-s0"}
EXTRA=${EXTRA:-}
TAG=${TAG:-P2H10-F-s0}
IMAGE=${IMAGE:-jev-alpasim}
COMMIT=0bb4c4bfe10951ea5589aa8ea514e422cf3c3506                       # the challenge-deployed AlpaSim commit (docs/alpasim.md)
git_hash=$(git -C "$repo" rev-parse --short HEAD)$(git -C "$repo" diff --quiet -- experiments/alpasim jevdrive lib experiments/op_parity experiments/op_adapt_l experiments/op_adapt_r2 || echo -dirty)
VCONT=${VCONT:-0}; LEAD=${LEAD:-0}
sw=; [[ $VCONT != 0 ]] && sw+=-vc$VCONT; [[ $LEAD == 1 ]] && sw+=-lead
VERSION=${VERSION:-$(echo "$TAG" | tr '+A-Z' '-a-z')-$git_hash$sw}
[[ $(git -C "$SRC" rev-parse HEAD) == "$COMMIT" ]] || { echo "AlpaSim checkout $SRC is not at $COMMIT" >&2; exit 2; }

ctx=$(mktemp -d "${TMPDIR:-/tmp}/jev-alpasim-ctx.XXXXXX"); trap 'rm -rf "$ctx"' EXIT
cp "$here"/{Dockerfile,requirements.lock,serve.py,probe.py,verify.py} "$ctx/"
S=e2e_challenge/sample_submission_simscale_navsim_transfuser/navsim_transfuser_challenge
mkdir -p "$ctx/jev-drive" "$ctx/alpasim" "$ctx/models/openpilot" "$ctx/ckpt" "$ctx/ckpt-extra"
(cd "$repo" && grep -v '^#' "$here/closure.txt" | xargs cp --parents -t "$ctx/jev-drive")
git -C "$SRC" archive "$COMMIT" src/grpc LICENSE $S/__init__.py $S/navigation.py $S/trajectory.py | tar -x -C "$ctx/alpasim"   # as committed
cp "$DATA_DIR/models/openpilot/cinque.ort.onnx" "$ctx/models/openpilot/"
for t in $TAGS;  do mkdir "$ctx/ckpt/$t";       cp "$DATA_DIR/runs/op_parity/runs/$t/ckpt-final.pt" "$ctx/ckpt/$t/"; done
for t in $EXTRA; do mkdir "$ctx/ckpt-extra/$t"; cp "$DATA_DIR/runs/op_parity/runs/$t/ckpt-final.pt" "$ctx/ckpt-extra/$t/"; done
touch "$ctx/ckpt-extra/.keep"; chmod -R a+rX,go-w "$ctx"
(cd "$ctx" && find . -type f ! -name Dockerfile -exec sha256sum {} + | sort -k2) > "$ctx/MANIFEST.sha256"
echo "context: $(du -sh "$ctx" | cut -f1), $(wc -l < "$ctx/MANIFEST.sha256") files; image $IMAGE:$VERSION (default $TAG, repo $git_hash)"

args=(--build-arg "TAG=$TAG" --build-arg "DRIVER=${DRIVER:-auto}" --build-arg "GIT_HASH=$git_hash"
      --build-arg "CUDA=${CUDA:-cu126}" --build-arg "VCONT=$VCONT" --build-arg "LEAD=$LEAD")
[[ -z ${WHEELS+x} && -d ${DATA_DIR:-}/cache/jev_wheels ]] && WHEELS=$DATA_DIR/cache/jev_wheels
if [[ -n ${WHEELS:-} ]]; then args+=(--build-context "wheels=$WHEELS" --build-arg "UV_SRC=--no-index --find-links /wheels")
else mkdir "$ctx/.nowheels"; args+=(--build-context "wheels=$ctx/.nowheels"); fi
[[ -n ${PROXY:-} ]] && args+=(--network host --build-arg "HTTP_PROXY=$PROXY" --build-arg "HTTPS_PROXY=$PROXY" --build-arg "NO_PROXY=localhost,127.0.0.1")
for try in 1 2 3 4 5 6; do                                             # the link drops; finished layers stay cached between tries
  DOCKER_BUILDKIT=1 docker build "${args[@]}" -t "$IMAGE:$VERSION" "$ctx" && break
  (( try < 6 )) || { echo "build failed after $try tries" >&2; exit 1; }
  echo "build try $try failed, retrying in 20 s"; sleep 20
done
out=${OUT:-}; [[ -n $out ]] && { mkdir -p "$out"; cp "$ctx/MANIFEST.sha256" "$out/"; docker history --no-trunc --format '{{.Size}}\t{{.CreatedBy}}' "$IMAGE:$VERSION" | cut -c1-220 > "$out/layers.txt"
                                 docker image inspect "$IMAGE:$VERSION" --format '{{.Id}} {{.Size}}' > "$out/image.txt"; echo "$IMAGE:$VERSION" > "$out/IMAGE"; }
docker image inspect "$IMAGE:$VERSION" --format "built $IMAGE:$VERSION {{.Id}} size {{.Size}} bytes"
