#!/usr/bin/env bash
# Download the NuRec scenes of a pai_scenes.py list into the wizard's scene cache layout, restartable. Runs ON the Tokyo box.
#   pai_fetch.sh <scenes.tsv> [max stage, default 1] [parallel, default 4]
# Each scene is one .usdz of nvidia/PhysicalAI-Autonomous-Vehicles-NuRec (gated; the token is read from ~/.cache/huggingface/token and
# never printed) -> $NUREC/all-usdzs/<uuid>.usdz, the name the wizard looks for (alpasim_wizard/scenes/sceneset.py), after the same
# check it makes: metadata.yaml inside the archive carries the catalogue uuid. Each scene is fetched in PAI_RANGES (default 8) ranged
# streams (ranged_get.sh: resume, bytes appended only on HTTP 206, one writer per scene). On the GPU box use HFROOT=https://hf-mirror.com
# without a proxy (huggingface.co through Clash gave 1.7 MB/s in total); PAI_PROXY routes through Clash. The Clash node is not rotated here.
# Log: $NUREC/fetch.log, one line per scene.
set -uo pipefail
LIST=$1 STAGE=${2:-1} PAR=${3:-4}
export NUREC=${NUREC:-${DATA_DIR:-/data}/datasets/nurec}
export HFREPO=${HFROOT:-https://huggingface.co}/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/resolve
. "$(dirname "$0")/ranged_get.sh"; export -f ranged_get
mkdir -p "$NUREC/all-usdzs" "$NUREC/_dl"
one() {  # uuid path revision
  local uuid=$1 path=$2 rev=$3 f=$NUREC/_dl/$1.usdz out=$NUREC/all-usdzs/$1.usdz t0=$SECONDS got size
  [[ -s $out ]] && return 0
  size=$(curl -sIL ${PAI_PROXY:+-x "$PAI_PROXY"} -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" "$HFREPO/$rev/$path" | tr -d '\r' \
         | awk 'tolower($1)=="content-length:"{s=$2} END{print s}')
  [[ $size -gt 1000000 ]] || { echo "$(date '+%F %T') FAILED $uuid: no size" >> "$NUREC/fetch.log"; return 1; }
  RG_PROXY=${PAI_PROXY:-} ranged_get "$HFREPO/$rev/$path" "$f" "$size" "${PAI_RANGES:-8}" 'echo "Authorization: Bearer $(cat ~/.cache/huggingface/token)"' \
    || { echo "$(date '+%F %T') FAILED $uuid" >> "$NUREC/fetch.log"; return 1; }
  got=$(unzip -p "$f" metadata.yaml 2>/dev/null | sed -n 's/^uuid: *//p' | tr -d "\"'")
  [[ $got == "$uuid" ]] || { echo "$(date '+%F %T') BAD $uuid: metadata uuid '$got'" >> "$NUREC/fetch.log"; return 1; }
  mv "$f" "$out"
  echo "$(date '+%F %T') ok $uuid $(stat -c%s "$out") bytes $((SECONDS - t0)) s" >> "$NUREC/fetch.log"
}
export -f one
awk -F'\t' -v s="$STAGE" 'NR > 1 && $8 <= s {print $2, $3, $4}' "$LIST" | xargs -P "$PAR" -L 1 bash -c 'one "$@"' _
rc=$?
echo "$(date '+%F %T') pass done rc=$rc: $(ls "$NUREC/all-usdzs" | wc -l) scenes, $(du -sh "$NUREC/all-usdzs" | cut -f1)" | tee -a "$NUREC/fetch.log"
exit $rc
