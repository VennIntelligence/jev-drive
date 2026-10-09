#!/usr/bin/env bash
# Unpack the PAI renderer image nvcr.io/nvidia/nre/nre-ga:26.04 into a plain directory, without Docker or root: the registry
# allows anonymous pulls (token from nvcr.io/proxy_auth), every layer is fetched with resume + one flock per layer, sha256 checked,
# and extracted in order into $NRE_ROOT (whiteouts applied). Runs on the GPU box through its Clash proxy.
#   pai_nre_pull.sh            -> $DATA_DIR/tools/nre-rootfs ; STATUS / DONE / ERROR in $DATA_DIR/runs/alpasim/pai2/nre_pull
set -uo pipefail
D=${DATA_DIR:-/data}; ROOT=${NRE_ROOT:-$D/tools/nre-rootfs}; OUT=$D/runs/alpasim/pai2/nre_pull; DL=$OUT/layers
IMG=${NRE_IMG:-nvidia/nre/nre-ga} TAGV=${NRE_TAG:-26.04}; REG=${NRE_REG:-nvcr.m.daocloud.io} PROXY=${PAI_PROXY:-}   # DaoCloud's nvcr.io mirror, direct: 1.6-4 MB/s; nvcr.io through Clash gave 0.1 MB/s (2026-10-09)
source "$(dirname "$0")/ranged_get.sh"; mkdir -p "$DL" "$ROOT"; rm -f "$OUT/DONE" "$OUT/ERROR"; exec > >(tee -a "$OUT/log.txt") 2>&1
die() { echo "ERROR $*" | tee "$OUT/ERROR"; exit 1; }
tok() { curl -sf ${PROXY:+-x "$PROXY"} -m 30 "https://m.daocloud.io/auth/token?service=$REG&scope=repository:$IMG:pull" | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])'; }
# the token endpoint is flaky through the proxy and a token lives 10 min: a background loop keeps a fresh one in a file every connection reads
TF=$OUT/token; until T=$(tok) && [[ -n $T ]]; do sleep 3; done; echo "$T" > "$TF"
( while sleep 200; do t=$(tok) && [[ -n $t ]] && echo "$t" > "$TF.new" && mv "$TF.new" "$TF"; done ) & TOKPID=$!; trap 'kill $TOKPID' EXIT
[[ -s $OUT/manifest.json ]] || until curl -sf ${PROXY:+-x "$PROXY"} -m 30 -H "Authorization: Bearer $T" -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
  "https://$REG/v2/$IMG/manifests/$TAGV" -o "$OUT/manifest.json"; do sleep 3; done
mapfile -t L < <(python3 -c 'import json;[print(l["digest"],l["size"]) for l in json.load(open("'"$OUT"'/manifest.json"))["layers"]]')
echo "${#L[@]} layers"; i=0
for e in "${L[@]}"; do
  d=${e% *}; sz=${e#* }; i=$((i + 1)); f=$DL/${d#sha256:}.tgz
  [[ -e $DL/${d#sha256:}.extracted ]] && continue
  RG_PROXY=$PROXY ranged_get "https://$REG/v2/$IMG/blobs/$d" "$f" "$sz" "$(( sz > 2000000000 ? 24 : sz > 400000000 ? 16 : sz > 20000000 ? 8 : 1 ))" 'echo "Authorization: Bearer $(cat "$TF")"' || die "layer $i download"
  [[ $(sha256sum "$f" | cut -d' ' -f1) == "${d#sha256:}" ]] || { rm -f "$f"; die "layer $i sha256"; }
  # extract; whiteouts: .wh.<name> removes <name>, .wh..wh..opq clears the directory's older content
  tar -tzf "$f" 2>/dev/null | grep -E '(^|/)\.wh\.' | while read -r w; do
    dir=$(dirname "$w"); b=$(basename "$w")
    if [[ $b == .wh..wh..opq ]]; then find "$ROOT/$dir" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null; else rm -rf "$ROOT/$dir/${b#.wh.}"; fi
  done
  tar -xzf "$f" -C "$ROOT" --no-same-owner --exclude='.wh.*' --delay-directory-restore 2>>"$OUT/tar_err.txt"
  chmod -R u+rwX "$ROOT" 2>/dev/null; touch "$DL/${d#sha256:}.extracted"; rm -f "$f"
  echo "$(date '+%F %T') layer $i/${#L[@]} ok $sz bytes" | tee "$OUT/STATUS"
done
du -sh "$ROOT" | tee -a "$OUT/log.txt"; date > "$OUT/DONE"
