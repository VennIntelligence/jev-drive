#!/usr/bin/env bash
# PAI curated-validation read of one checkpoint on the GPU box: pai_eval.sh's chunked read with one GPU-pool job per chunk instead of one
# Docker stack per Tokyo card. Each job is pai_native.sh (no containers: unpacked NRE renderer, every service a host process, our driver
# from envs/op-train); the pool places the jobs, so a read spreads over every card with room (a stack peaks at 26-27 GB, a box card holds three).
#   pai_box.sh submit <out root> <checkpoint tag> <scene list>...   lists: pai_scenes.py tsv (stage <= 1 rows) or plain scene ids
#   pai_box.sh run <out root> <checkpoint tag> <chunk>              job body (one chunk)
# Env (submit): CHUNK (scenes per job, 25), CHUNKS_FROM (a dir of chunk lists, e.g. a Tokyo read's chunks/: the simulator repeats a scene
#   only within an identical list, so a parity read reuses the other read's lists), CONC (4 rollouts per stack), SERVE (driver switches,
#   default the served configuration "JEV_VCONT=1.0 JEV_LEAD=1"; other driver env such as JEV_VHIST=holdlead or SH30_RMEM=0 goes here too),
#   WIZ (wizard overrides, space separated, e.g. "controller.gains.rel_front_steering_angle_weight=20"), PRIO (pool priority, 13),
#   AFTER (pool job ids the chunks wait for), KEEP_VIDEO=1 (keep the rollout mp4s; dropped by default, 0.2-0.3 GB per scene).
# Layout as pai_eval.sh, so pai_eval_report.py reads it unchanged: <out>/{chunks/cNNN.txt, jobs.txt, runs/<lowercase tag>_cNNN/{sim/, driver ->
#   sim/driver-logs, pool/, SUBMITTED, DONE | ERROR}}. A second submit skips finished and pending chunks and resubmits failed ones (resume);
#   a chunk with a scene not yet in the scene cache is left out until a later submit.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); repo=$(cd "$here/../../.." && pwd)
mode=$1 O=$(mkdir -p "$2" && cd "$2" && pwd) T=$3; shift 3
C=$DATA_DIR/runs/op_parity/runs/$T; L=${T,,}
case $mode in
submit)
  [[ -f $C/ckpt-final.pt ]] || { echo "no checkpoint $C" >&2; exit 1; }
  mkdir -p "$O"/{chunks,runs}
  if ! ls "$O"/chunks/*.txt >/dev/null 2>&1; then              # written once: a resubmit keeps its chunks
    if [[ -n ${CHUNKS_FROM:-} ]]; then cp "$CHUNKS_FROM"/*.txt "$O/chunks/"
    else for f in "$@"; do if head -1 "$f" | grep -q '^scene_id'; then awk -F'\t' 'NR > 1 && $8 <= 1 {print $1}' "$f"; else grep . "$f"; fi; done \
           | awk '!seen[$0]++' | split -l "${CHUNK:-25}" -d -a 3 --additional-suffix=.txt - "$O/chunks/c"; fi
  fi
  printf '%s\n' "SERVE=${SERVE-JEV_VCONT=1.0 JEV_LEAD=1}" "WIZ=${WIZ:-}" "CONC=${CONC:-4}" "KEEP_VIDEO=${KEEP_VIDEO:-0}" "code=$(git -C "$repo" rev-parse --short HEAD)" > "$O/config.txt"
  X=(); for kv in ${SERVE-JEV_VCONT=1.0 JEV_LEAD=1}; do X+=(--env "$kv"); done
  miss=$(python3 - "$DATA_DIR/third_party/alpasim/data/scenes" "${NUREC:-$DATA_DIR/datasets/nurec}/all-usdzs" "$O"/chunks/*.txt <<'PY'
import csv, os, sys                 # chunks with a scene whose usdz is not in the cache yet (a copy still running): left for a later submit
src, cache, *chunks = sys.argv[1:]; have = set(os.listdir(cache)); ok = set()
for f in ("sim_scenes.csv", "sim_scenes_2604.csv"):
    ok |= {r["scene_id"] for r in csv.DictReader(open(f"{src}/{f}")) if r["uuid"] + ".usdz" in have}
print(" ".join(os.path.basename(c)[:-4] for c in chunks if any(s not in ok for s in open(c).read().split())))
PY
)
  for f in "$O"/chunks/*.txt; do
    c=$(basename "$f" .txt); r=$O/runs/${L}_$c
    [[ -f $r/DONE ]] && continue
    [[ -f $r/SUBMITTED && ! -f $r/ERROR ]] && continue                # queued or running
    [[ " $miss " == *" $c "* ]] && { echo "$c: scenes missing from the cache, not submitted"; continue; }
    mkdir -p "$r/pool"
    id=$(cd "$repo" && .venv/bin/python -m jevdrive.cl submit --no-check --owner pai-box --name "pai-$L-$c" --vram 28 --cpu 6 --ram 40 --timeout-h 3 \
           --tries 2 --priority "${PRIO:-13}" ${AFTER:+--after "$AFTER"} --log-dir "$r/pool" --env CL_HIST_KEY=pai-read4 --env "CONC=${CONC:-4}" \
           --env "WIZ=${WIZ:-}" --env "KEEP_VIDEO=${KEEP_VIDEO:-0}" "${X[@]}" -- bash "$here/pai_box.sh" run "$O" "$T" "$c" | awk '{print $NF}')
    [[ -n $id ]] || { echo "submit $c failed" >&2; exit 1; }
    echo "$id" > "$r/SUBMITTED"; echo "$id pai-$L-$c" | tee -a "$O/jobs.txt"
  done ;;
run)
  c=$1; r=$O/runs/${L}_$c; R=$r/sim
  rm -rf "$R" "$r"/{DONE,ERROR}; mkdir -p "$r"; ln -sfn sim/driver-logs "$r/driver"
  # shellcheck disable=SC2086
  env SH30_TAG="$T" CONC="${CONC:-4}" bash "$here/pai_native.sh" "$R" "$O/chunks/$c.txt" ${WIZ:-}; rc=$?
  rm -rf "$R/nre-cache" "$R/driver-tmp"; [[ ${KEEP_VIDEO:-0} == 1 ]] || find "$R/rollouts" -name '*.mp4' -delete 2>/dev/null
  if (( rc == 0 )) && [[ -f $R/aggregate/results-summary.json ]]; then date > "$r/DONE"; else echo "rc $rc" > "$r/ERROR"; exit 1; fi ;;
*) sed -n '2,14p' "$0"; exit 2 ;;
esac
