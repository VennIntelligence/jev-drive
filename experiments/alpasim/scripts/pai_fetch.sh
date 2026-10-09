#!/usr/bin/env bash
# Download the NuRec scenes of a pai_scenes.py list into the wizard's scene cache layout, restartable. Runs ON the Tokyo box.
#   pai_fetch.sh <scenes.tsv> [max stage, default 1] [parallel, default 4]
# Each scene is one .usdz of nvidia/PhysicalAI-Autonomous-Vehicles-NuRec (gated; the token is read from ~/.cache/huggingface/token and
# never printed) -> $NUREC/all-usdzs/<uuid>.usdz, the name the wizard looks for (alpasim_wizard/scenes/sceneset.py), after the same
# check it makes: metadata.yaml inside the archive carries the catalogue uuid. A stalled connection (< 200 kB/s for 30 s) is cut and
# resumed; the Clash node is not rotated here (another session shares the link). Log: $NUREC/fetch.log, one line per scene.
set -uo pipefail
LIST=$1 STAGE=${2:-1} PAR=${3:-4}
export NUREC=${NUREC:-${DATA_DIR:-/data}/datasets/nurec}
export HFREPO=https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/resolve
mkdir -p "$NUREC/all-usdzs" "$NUREC/_dl"
one() {  # uuid path revision
  local uuid=$1 path=$2 rev=$3 f=$NUREC/_dl/$1.part out=$NUREC/all-usdzs/$1.usdz t0=$SECONDS n=0 got
  [[ -s $out ]] && return 0
  until curl -sfL -C - --speed-limit 200000 --speed-time 30 -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" \
      -o "$f" "$HFREPO/$rev/$path"; do
    n=$((n + 1)); (( n > 200 )) && { echo "$(date '+%F %T') FAILED $uuid after $n tries" >> "$NUREC/fetch.log"; return 1; }; sleep 3
  done
  got=$(unzip -p "$f" metadata.yaml 2>/dev/null | sed -n 's/^uuid: *//p' | tr -d "\"'")
  [[ $got == "$uuid" ]] || { echo "$(date '+%F %T') BAD $uuid: metadata uuid '$got'" >> "$NUREC/fetch.log"; return 1; }
  mv "$f" "$out"
  echo "$(date '+%F %T') ok $uuid $(stat -c%s "$out") bytes $((SECONDS - t0)) s $n retries" >> "$NUREC/fetch.log"
}
export -f one
awk -F'\t' -v s="$STAGE" 'NR > 1 && $8 <= s {print $2, $3, $4}' "$LIST" | xargs -P "$PAR" -L 1 bash -c 'one "$@"' _
rc=$?
echo "$(date '+%F %T') pass done rc=$rc: $(ls "$NUREC/all-usdzs" | wc -l) scenes, $(du -sh "$NUREC/all-usdzs" | cut -f1)" | tee -a "$NUREC/fetch.log"
exit $rc
