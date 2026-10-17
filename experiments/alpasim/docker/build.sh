#!/usr/bin/env bash
# Build the AlpaSim submission image (Dockerfile here) on a Docker host that holds the repo, the AlpaSim checkout and the weights at the
# box's relative paths under $DATA_DIR (Tokyo: /data). Stages a minimal context (closure.txt, src/grpc, two LTF-sample helpers,
# cinque.ort.onnx, one ckpt-final.pt per tag + its route_mem.json sidecar if any), builds, prints size and layers. Nothing else of the repo enters the image.
#   experiments/alpasim/docker/build.sh                               # default: serves P2H10-F-s0
#   TAG=AP2-AB-s0 experiments/alpasim/docker/build.sh                 # the one-line switch: same layers, another default
#   EXTRA="OT10a05-F-s0" TAG=OT10a05-F-s0 .../build.sh                # a new checkpoint: one thin layer on top of the cached ones
#   TRACK=pai TAG=RSG-L18-s0 .../build.sh                             # PAI: pai_driver.py, JEV_VCONT=1.0 JEV_LEAD=1, JEV_TRACK unset (v8 gate)
#   TRACK=pai TAG=RSG-L18-s0 VHIST=holdlead ACC=1 .../build.sh        # PAI with the optional switches: same layers, another last layer
#   TRACK=pai TAG=RSG-L18-s0 ACC=1 GAP=0.7,2.7 .../build.sh           # + the lead MPC following gap (JEV_GAP, needs LEAD=1): one more ENV layer
# Env: TRACK (nuplan | pai, default nuplan), TAGS (stable checkpoint layer; pai: "RSG-L18-s0 RBSG2-L18-s0"), EXTRA (thin layer), TAG (served
# by default; several joined by + = ensemble, nuPlan only), DRIVER (auto | sh30 | ap2 | ens | pai; pai forces pai), VCONT / LEAD / ACC / VHIST
# (the serving switches baked in as JEV_VCONT / JEV_LEAD / JEV_ACC / JEV_VHIST, lib/serve_fix.py, lib/vhist.py; nuplan default 0 / 0 / 0 / ""
# = off, pai default 1.0 / 1 / 0 / "" = the box reads' served configuration; JEV_TRACK is never baked in), GAP (JEV_GAP=<t_follow>[,<stop m>],
# default "" = off; its own ENV layer after the one above, needs LEAD=1),
# IMAGE (repository, default jev-alpasim), VERSION (image tag, default <TAG>-<git short hash>[-pai][-vc<VCONT>][-lead][-vh<VHIST>][-acc][-gap<GAP, comma -> _>]),
# WHEELS (dir of pre-fetched wheels, see wheels.py: the build then installs offline from it; default $DATA_DIR/cache/jev_wheels when it
# exists, WHEELS= forces the indexes, whose triton wheel does not match requirements.lock as of 2026-10-09), PROXY (http proxy for the build's
# downloads, e.g. http://127.0.0.1:7890; uses the host network), DATA_DIR, ALPASIM_SRC.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); repo=$(cd "$here/../../.." && pwd)
: "${DATA_DIR:?DATA_DIR}"
SRC=${ALPASIM_SRC:-$DATA_DIR/third_party/alpasim}
TRACK=${TRACK:-nuplan}
case $TRACK in
  nuplan) TAGS=${TAGS:-"P2H10-F-s0 AP2-AB-s0"}; TAG=${TAG:-P2H10-F-s0}; VCONT=${VCONT:-0}; LEAD=${LEAD:-0} ;;
  pai)    TAGS=${TAGS:-"RSG-L18-s0 RBSG2-L18-s0"}; TAG=${TAG:-RSG-L18-s0}; VCONT=${VCONT:-1.0}; LEAD=${LEAD:-1}; DRIVER=pai
          [[ $TAG != *+* ]] || { echo "TRACK=pai serves one checkpoint" >&2; exit 2; } ;;
  *) echo "TRACK=$TRACK: nuplan | pai" >&2; exit 2 ;;
esac
EXTRA=${EXTRA:-}
IMAGE=${IMAGE:-jev-alpasim}
COMMIT=0bb4c4bfe10951ea5589aa8ea514e422cf3c3506                       # the challenge-deployed AlpaSim commit (docs/alpasim.md)
git_hash=$(git -C "$repo" rev-parse --short HEAD)$(git -C "$repo" diff --quiet -- experiments/alpasim jevdrive lib experiments/op_parity experiments/op_adapt_l experiments/op_adapt_r2 experiments/route_mem || echo -dirty)
ACC=${ACC:-0}; VHIST=${VHIST:-}; GAP=${GAP:-}
[[ -z $GAP || $LEAD == 1 ]] || { echo "GAP needs LEAD=1" >&2; exit 2; }
sw=; [[ $TRACK == pai ]] && sw+=-pai; [[ $VCONT != 0 ]] && sw+=-vc$VCONT; [[ $LEAD == 1 ]] && sw+=-lead
[[ -n $VHIST && $VHIST != off ]] && sw+=-vh$VHIST; [[ $ACC == 1 ]] && sw+=-acc; [[ -n $GAP ]] && sw+=-gap${GAP//,/_}
VERSION=${VERSION:-$(echo "$TAG" | tr '+A-Z' '-a-z')-$git_hash$sw}
[[ $(git -C "$SRC" rev-parse HEAD) == "$COMMIT" ]] || { echo "AlpaSim checkout $SRC is not at $COMMIT" >&2; exit 2; }

ctx=$(mktemp -d "${TMPDIR:-/tmp}/jev-alpasim-ctx.XXXXXX"); trap 'rm -rf "$ctx"' EXIT
cp "$here"/{Dockerfile,requirements.lock,serve.py,probe.py,verify.py,pai_spec.json} "$ctx/"
S=e2e_challenge/sample_submission_simscale_navsim_transfuser/navsim_transfuser_challenge
mkdir -p "$ctx/jev-drive" "$ctx/alpasim" "$ctx/models/openpilot" "$ctx/ckpt" "$ctx/ckpt-extra"
(cd "$repo" && grep -v '^#' "$here/closure.txt" | xargs cp --parents -t "$ctx/jev-drive")
git -C "$SRC" archive "$COMMIT" src/grpc LICENSE $S/__init__.py $S/navigation.py $S/trajectory.py | tar -x -C "$ctx/alpasim"   # as committed
cp "$DATA_DIR/models/openpilot/cinque.ort.onnx" "$ctx/models/openpilot/"
cpk() { mkdir "$2/$1"; cp "$DATA_DIR/runs/op_parity/runs/$1/ckpt-final.pt" "$2/$1/"     # + the ROUTE-MEM sidecar, read by pai_driver / sh30_driver
        [[ -f $DATA_DIR/runs/op_parity/runs/$1/route_mem.json ]] && cp "$DATA_DIR/runs/op_parity/runs/$1/route_mem.json" "$2/$1/"; return 0; }
for t in $TAGS;  do cpk "$t" "$ctx/ckpt"; done
for t in $EXTRA; do cpk "$t" "$ctx/ckpt-extra"; done
touch "$ctx/ckpt-extra/.keep"; chmod -R a+rX,go-w "$ctx"
(cd "$ctx" && find . -type f ! -name Dockerfile -exec sha256sum {} + | sort -k2) > "$ctx/MANIFEST.sha256"
echo "context: $(du -sh "$ctx" | cut -f1), $(wc -l < "$ctx/MANIFEST.sha256") files; image $IMAGE:$VERSION (default $TAG, repo $git_hash)"

args=(--build-arg "TAG=$TAG" --build-arg "DRIVER=${DRIVER:-auto}" --build-arg "GIT_HASH=$git_hash"
      --build-arg "CUDA=${CUDA:-cu126}" --build-arg "VCONT=$VCONT" --build-arg "LEAD=$LEAD" --build-arg "ACC=$ACC" --build-arg "VHIST=$VHIST")
[[ -n $GAP ]] && args+=(--build-arg "GAP=$GAP")
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
