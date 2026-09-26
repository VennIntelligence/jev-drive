#!/usr/bin/env bash
# Night queue 4, P3 (todos/2026-09-26-night-queue-4.md, P section, [P3] entries): real-appearance pedestrian exam,
# feasibility on 10 WOD scene-flow segments with drivestudio OmniRe (no SMPL; pedestrians as DeformableNodes).
#   install -> scenes -> prep (all 10, CPU) -> scene 0 smoke on the debug card (sky, train, render) -> index -> op
#   -> exam -> report -> SMOKE (checklist in STATUS.md) -> wait for GO -> scenes 1-9 on the granted cards
#   (sky, train, render per scene; WORKERS scenes per card) -> index -> op -> exam -> report -> DONE
# GO ($D/GO, written by SCH or Codex once scene 0 passes): shell vars GPUS="a b ..", WORKERS=<scenes per card>, P3_CPUS=<list>.
# Idempotent: a step with its .done marker is skipped. A step is retried once; a second failure writes ERROR and stops.
# Every step runs under timeout = 2x its estimate. Cores: $P3_CPUS (default 60-67), threads capped to match.
# Start: scripts/tmux_run.sh nq4-p3 scripts/nq4_p3.sh
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
REPO=$PWD
D=$DATA_DIR/runs/nq4/p3
mkdir -p "$D"
exec 9>"$D/lock"
flock -n 9 || { echo "nq4-p3: another chain holds $D/lock"; exit 1; }
[[ -f $D/DONE ]] && { echo "nq4-p3: DONE already: $(cat "$D/DONE")"; exit 0; }
rm -f "$D/ERROR"
echo $$ > "$D/chain.pid"
DEBUG_GPU=${P3_DEBUG_GPU:-1}
CPUS=${P3_CPUS:-60-67}
NT=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT \
       TF_NUM_INTRAOP_THREADS=$NT TF_NUM_INTEROP_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
JV=$DATA_DIR/envs/jevdrive/bin/python
DSPY=$DATA_DIR/envs/drivestudio/bin/python
PREP=$DATA_DIR/envs/p3-wodprep/bin/python
OPPY=$DATA_DIR/envs/openpilot/bin/python
RAW=$DATA_DIR/datasets/waymo_perception/sceneflow
PROC=$DATA_DIR/processed/waymo_ds          # drivestudio layout: $PROC/training/<k 03d>
RUNS=$DATA_DIR/ckpt/nq4_p3                  # OmniRe runs: $RUNS/p3/<k 03d>
SC=$DATA_DIR/processed/nq4_p3/scenes
EST_TRAIN=$(cat "$D/est_train_min" 2>/dev/null || echo 120)
declare -A EST=([install]=60 [scenes]=2 [prep]=45 [sky]=10 [train]=$EST_TRAIN [render]=15 [index]=5 [op]=20 [exam]=20 [report]=5)

ev() { printf '{"t": "%s", "event": "%s", %s}\n' "$(date '+%F %T')" "$1" "${2:-\"x\": 0}" >> "$D/events.jsonl"; }
log() { echo "$(date '+%F %H:%M:%S') $*" | tee -a "$D/chain.log"; }

status() {
    { echo "# nq4 P3 chain ($(date '+%F %H:%M:%S'))"; echo
      echo "pid $$ (also $D/chain.pid), cores $CPUS, debug card $DEBUG_GPU; train estimate $EST_TRAIN min per scene (2x = stop)"
      echo; for f in $(ls "$D"/*.done 2>/dev/null | sort); do echo "- $(basename "$f" .done): done $(cat "$f")"; done
      [[ -n ${1:-} ]] && echo "- $1: RUNNING since $(date +%H:%M) ${2:-}"
      [[ -f $D/smoke_checklist.md ]] && { echo; cat "$D/smoke_checklist.md"; }
      if [[ -f $D/SMOKE && ! -f $D/GO ]]; then echo; echo "**Waiting for $D/GO** (GPUS / WORKERS / P3_CPUS) to run scenes 1-9."; fi
      echo; echo "logs: $D/*.out, $D/chain.log, $D/events.jsonl; runs $RUNS/p3/<scene>; renders $SC"; } > "$D/STATUS.md"
}

fail() { log "ERROR: $*"; echo "$*" > "$D/ERROR"; ev error "\"msg\": \"$*\""; status; exit 1; }

# step <name> <est key> <gpu or -> <cmd...>: skip if done, retry once, timeout at 2x the estimate
step() {
    local name=$1 key=$2 gpu=$3; shift 3
    [[ -f $D/$name.done ]] && return 0
    local attempt rc t0 el
    for attempt in 1 2; do
        status "$name" "(GPU $gpu, attempt $attempt)"
        log "step $name attempt $attempt GPU $gpu: $*"; ev start "\"step\": \"$name\", \"gpu\": \"$gpu\", \"attempt\": $attempt"
        t0=$(date +%s)
        CUDA_VISIBLE_DEVICES=$gpu timeout $(( EST[$key] * 2 * 60 )) taskset -c "$CPUS" "$@" >> "$D/$name.out" 2>&1
        rc=$?; el=$(( ($(date +%s) - t0) / 60 ))
        if (( rc == 0 )); then
            echo "$(date '+%H:%M') after ${el} min on GPU $gpu" > "$D/$name.done"
            ev done "\"step\": \"$name\", \"min\": $el"; return 0
        fi
        log "step $name attempt $attempt failed rc=$rc after ${el} min (124 = over 2x the estimate $((EST[$key] * 2)) min)"
    done
    return 1
}

scene_job() {   # scene_job <k> <gpu>: sky, train, render of one scene
    local k=$1 g=$2 kk
    kk=$(printf %03d "$1")
    step "sky_$kk" sky "$g" $DSPY scripts/p3/ds.py sky "$PROC/training/$kk" || return 1
    step "train_$kk" train "$g" $DSPY scripts/p3/ds.py train --scene "$k" --data-root "$PROC/training" --out-root "$RUNS" || return 1
    step "render_$kk" render "$g" $DSPY scripts/p3/ds.py render --run "$RUNS/p3/$kk" --target "$D/targets/$kk.json" \
        --out "$SC/p3_$kk" || return 1
}

readout() {    # readout <tag>: index -> openpilot features -> exam -> report over every rendered scene
    local tag=$1
    step "index_$tag" index - $JV -m jevdrive.nq4_p3 index --processed-root "$PROC/training" || return 1
    if [[ ! -f $D/op_$tag.done ]]; then rm -rf "$DATA_DIR/processed/nq4_p3/op_streams" "$DATA_DIR"/processed/nq4_p3/op_cinque "$DATA_DIR"/processed/nq4_p3/op_lebowski; fi
    step "op_$tag" op "$DEBUG_GPU_OR_BATCH" env P5_SET=nq4_p3 $OPPY scripts/p5_openpilot.py --models cinque lebowski --workers 6 || return 1
    step "opfin_$tag" index - env P5_SET=nq4_p3 $JV -m jevdrive.p5_openpilot finalize --models cinque,lebowski || return 1
    step "exam_$tag" exam "$DEBUG_GPU_OR_BATCH" $JV -m jevdrive.nq4_p3 exam || return 1
    local ex
    ex=$(ls -d "$DATA_DIR"/runs/nq4/p3-exam/*/ 2>/dev/null | sort | tail -1)
    step "report_$tag" report - $JV -m jevdrive.nq4_p3 report --exam-dir "$ex" --out "$D/report_$tag" || return 1
}

# ---------------------------------------------------------------- setup and scene 0 smoke (debug card)
step install install - bash scripts/p3/install_drivestudio.sh || fail "install failed; see $D/install.out"
step scenes scenes - $JV -m jevdrive.nq4_p3 scenes || fail "scenes"
[[ $(ls "$RAW"/*.tfrecord 2>/dev/null | wc -l) -ge 10 ]] || fail "expected 10 scene-flow tfrecords in $RAW"
step prep prep - env CUDA_VISIBLE_DEVICES= $PREP "$REPO/scripts/p3/ds.py" prep --raw "$RAW" --out "$PROC" \
    --scenes "$D/scenes.json" --workers 8 || fail "prep (Waymo preprocess) failed; see $D/prep.out"
DEBUG_GPU_OR_BATCH=$DEBUG_GPU
if [[ ! -f $D/SMOKE ]]; then
    t0=$(date +%s)
    scene_job 0 "$DEBUG_GPU" || fail "scene 0 smoke failed; see $D/{sky,train,render}_000.out"
    readout smoke || fail "scene 0 readout failed; see $D/*_smoke.out"
    tr=$(awk '{print $3}' "$D/train_000.done"); echo $(( tr + tr / 2 + 5 )) > "$D/est_train_min"
    $JV - "$D" <<'PY' || fail "smoke checklist"
import json, sys, pandas as pd
from pathlib import Path
D = Path(sys.argv[1]); r = pd.read_csv(D / "report_smoke/scenes.csv").iloc[0]
g = pd.read_csv(D / "report_smoke/null_gate_pooled.csv").set_index("examinee")
c = g.loc["ridge_late op-cinque temporal"]
rows = [("reconstruction PSNR (plus vs real, full image)", f"{r.psnr:.2f} dB", r.psnr >= 25),
        ("PSNR inside pedestrian boxes", f"{r.psnr_ped:.2f} dB", True),
        ("render determinism (same view twice)", f"max |d| = {r.determinism_max_abs:.2e}", r.determinism_max_abs == 0),
        ("every corridor pedestrian found as a node", f"{int(r.n_deleted)} deleted, {int(r.deleted_missing)} missing", r.deleted_missing == 0),
        ("plus vs minus outside the deleted boxes (> 8/255)", f"{int(r.del_diff_px_out)} px", True),
        ("null false flip, ridge_late Cinque (tau_I3 %.2f)" % c.tau_i3, f"{c.null_false_flip:.3f} [{c.lo:.3f}, {c.hi:.3f}] over {int(c.n_null_frames)} frames", c.null_false_flip <= 0.07)]
md = ["## scene 0 smoke checklist (automatic part; residue is judged by eye from the figure, pending main review)", "",
      "| check | value | ok |", "|:--|:--|:--|"] + [f"| {a} | {b} | {'yes' if ok else 'NO'} |" for a, b, ok in rows]
(D / "smoke_checklist.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
PY
    echo "$(date '+%F %H:%M') scene 0 smoke finished in $(( ($(date +%s) - t0) / 60 )) min" > "$D/SMOKE"
    ev smoke "\"min\": $(( ($(date +%s) - t0) / 60 ))"
fi

# ---------------------------------------------------------------- GO gate
status
log "waiting for $D/GO"
until [[ -f $D/GO ]]; do sleep 60; done
# shellcheck disable=SC1091
source "$D/GO"
: "${GPUS:?GO must set GPUS}" "${WORKERS:=1}"
CPUS=${P3_CPUS:-$CPUS}
log "GO: GPUS=$GPUS WORKERS=$WORKERS CPUS=$CPUS"; ev go "\"gpus\": \"$GPUS\", \"workers\": $WORKERS"
EST[train]=$(cat "$D/est_train_min" 2>/dev/null || echo "${EST[train]}")
declare -A running=()
free_gpu() {   # a card in GPUS with fewer than WORKERS scene jobs alive, else empty
    local g p n
    for p in "${!running[@]}"; do kill -0 "$p" 2>/dev/null || unset "running[$p]"; done
    for g in $GPUS; do
        n=0; for p in "${!running[@]}"; do [[ ${running[$p]} == "$g" ]] && n=$((n + 1)); done
        (( n < WORKERS )) && { echo "$g"; return; }
    done
}
for k in $(seq 1 9); do
    [[ -f $D/render_$(printf %03d "$k").done ]] && continue
    until g=$(free_gpu) && [[ -n $g ]]; do sleep 20; done
    ( scene_job "$k" "$g" || echo "scene $k failed" >> "$D/scene_failures.txt" ) & running[$!]=$g
    log "scene $k -> GPU $g (pid $!)"
    sleep 5
done
wait
[[ -s $D/scene_failures.txt ]] && fail "$(cat "$D/scene_failures.txt" | tr '\n' ';')"
DEBUG_GPU_OR_BATCH=${GPUS%% *}
readout all || fail "final readout failed; see $D/*_all.out"
echo "$(date '+%F %H:%M') all 10 scenes rendered and read out; report in $D/report_all" > "$D/DONE"
ev done_all
status
log "DONE"
