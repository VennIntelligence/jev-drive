#!/usr/bin/env bash
# Unpack the PAI renderer image nvcr.io/nvidia/nre/nre-ga:26.04 into a plain directory, without Docker or root: the registry
# allows anonymous pulls (token from nvcr.io/proxy_auth), every layer is fetched with resume + one flock per layer, sha256 checked,
# and extracted in order into $NRE_ROOT (whiteouts applied). Runs on the GPU box through its Clash proxy.
#   pai_nre_pull.sh            -> $DATA_DIR/tools/nre-rootfs ; STATUS / DONE / ERROR in $DATA_DIR/runs/alpasim/pai2/nre_pull
set -uo pipefail
D=${DATA_DIR:-/data}; ROOT=${NRE_ROOT:-$D/tools/nre-rootfs}; OUT=$D/runs/alpasim/pai2/nre_pull; DL=$OUT/layers
IMG=${NRE_IMG:-nvidia/nre/nre-ga} TAGV=${NRE_TAG:-26.04}; PROXY=${PAI_PROXY:-http://127.0.0.1:7890}
mkdir -p "$DL" "$ROOT"; rm -f "$OUT/DONE" "$OUT/ERROR"; exec > >(tee -a "$OUT/log.txt") 2>&1
die() { echo "ERROR $*" | tee "$OUT/ERROR"; exit 1; }
tok() { curl -sf -x "$PROXY" -m 30 "https://nvcr.io/proxy_auth?service=registry&scope=repository:$IMG:pull" | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])'; }
T=$(tok) || die token
curl -sf -x "$PROXY" -m 30 -H "Authorization: Bearer $T" -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
  "https://nvcr.io/v2/$IMG/manifests/$TAGV" -o "$OUT/manifest.json" || die manifest
mapfile -t L < <(python3 -c 'import json;[print(l["digest"],l["size"]) for l in json.load(open("'"$OUT"'/manifest.json"))["layers"]]')
echo "${#L[@]} layers"; i=0
for e in "${L[@]}"; do
  d=${e% *}; sz=${e#* }; i=$((i + 1)); f=$DL/${d#sha256:}.tgz
  [[ -e $DL/${d#sha256:}.extracted ]] && continue
  exec {lk}>"$f.lock"; flock -n "$lk" || die "layer $i busy (another puller)"
  n=0; until [[ $(stat -c%s "$f" 2>/dev/null || echo 0) == "$sz" ]]; do
    (( ++n > 400 )) && die "layer $i too many retries"
    (( n % 20 == 0 )) && T=$(tok)
    have=$(stat -c%s "$f" 2>/dev/null || echo 0)
    code=$(curl -sL -x "$PROXY" -o "$f.new" -w '%{http_code}' --speed-limit 50000 --speed-time 60 --max-time 600 -H "Authorization: Bearer $T" \
      -r "$have-" "https://nvcr.io/v2/$IMG/blobs/$d" || true)
    [[ $code == 206 ]] && cat "$f.new" >> "$f"
    [[ $code == 200 && $have == 0 ]] && mv "$f.new" "$f"; rm -f "$f.new"; [[ $code == 206 || $code == 200 ]] || sleep 3
  done
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
