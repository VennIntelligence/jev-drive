#!/usr/bin/env bash
# Resumable download of one small VLM into the HF cache. Usage: small_vlm_dl.sh <repo> (run one per model, in parallel).
# JEV_DL_SRC=ms skips hf-mirror. Writes $OUT/<name>/{STATUS,DONE,ERROR}. Source order: hf-mirror, then ModelScope into $DATA_DIR/models/<name>.
set -uo pipefail
repo=$1; name=${repo#*/}
: "${DATA_DIR:?}"; : "${HF_HOME:?}"
out=$DATA_DIR/runs/small_vlm/dl/$name; mkdir -p "$out"; rm -f "$out/ERROR"
[[ -f $out/DONE ]] && exit 0
echo "start $(date -Is)" > "$out/STATUS"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com} HF_HUB_DISABLE_XET=1
for try in 1 2 3 4 5 6; do
  [[ ${JEV_DL_SRC:-hf} == ms ]] && break   # JEV_DL_SRC=ms: hf-mirror measured ~1 MB/s total, go straight to ModelScope
  echo "hf try $try $(date -Is)" >> "$out/STATUS"
  hf download "$repo" --max-workers 4 --exclude '.DS_Store' --exclude 'original/*' >> "$out/log.txt" 2>&1 && { echo ok > "$out/DONE"; echo "done $(date -Is)" >> "$out/STATUS"; exit 0; }
  sleep 10
done
echo "hf failed, trying ModelScope" >> "$out/STATUS"
for try in 1 2 3; do
  modelscope download "$repo" --local-dir "$DATA_DIR/models/$name" --exclude '.DS_Store' --max-workers 4 >> "$out/log.txt" 2>&1 \
    && { echo "modelscope:$DATA_DIR/models/$name" > "$out/DONE"; echo "done(ms) $(date -Is)" >> "$out/STATUS"; exit 0; }
  sleep 10
done
echo "all sources failed" > "$out/ERROR"
