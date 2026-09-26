#!/usr/bin/env bash
# NAVSIM reproduction of DrivoR (navtest PDMS, v1) and WA-JEPA (navtest EPDMS, v2) with each model's OWN
# evaluation entry points, run as shipped (todos/2026-09-26-top10-intersection.md, [T2] step 6).
# Only paths, env vars and worker counts are set here; no third-party file is modified.
#
#   drivor  DrivoR repo (navsim v1.1 fork): navsim/planning/script/run_pdm_score_multi_gpu.py with the README's
#           NAVSIM-v1 overrides, metric cache = our v1.1 navtest cache (same MetricCache class and scorer code)
#   wajepa  WA-JEPA repo: its trajectory export made resumable (wajepa_export.py: its NAVSIM agent, fp32, NPROC ranks on one GPU),
#           then eval.navsim_score_trajectory_cache on the v2 devkit (EPDMS, one-stage) and the v1.1 devkit (PDMS).
#           The v1.1 config already has sensor_blobs_path (same path), so the scorer's "+<key>=" append goes to an
#           unused key instead of failing on the existing one.
#
# Usage (on the box, in tmux): [RUN=<dir>] GPU=4 CPUS=0-3 [NPROC=3 SCORE_WORKERS=8] scripts/top10_t2/navsim_repro.sh drivor|wajepa [extra hydra overrides, drivor]
# CPUS pins the whole job (inference, data loading, scoring workers) to a core list: the box is CPU-bound.
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
model=${1:?drivor|wajepa}
GPU=${GPU:-4}
CPUS=${CPUS:-0-3}
pin=(taskset -c "$CPUS")
NPROC=${NPROC:-3}; SCORE_WORKERS=${SCORE_WORKERS:-8}
TP=$DATA_DIR/third_party
run=${RUN:-$DATA_DIR/runs/top10_t2/navsim/$model/$(date +%Y%m%d-%H%M%S)}   # RUN=<old run dir> resumes (wajepa)
mkdir -p "$run"
exec > >(tee -a "$run/log.txt") 2>&1
export OPENSCENE_DATA_ROOT=$DATA_DIR/datasets/navsim NUPLAN_MAPS_ROOT=$DATA_DIR/datasets/navsim/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0 OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
CACHE=$DATA_DIR/runs/navsim/metric_cache
echo "[$(date +%T)] $model -> $run (GPU $GPU, CPUs $CPUS)"

case $model in
  drivor)
    cd "$TP/drivor"
    export NAVSIM_DEVKIT_ROOT=$TP/drivor NAVSIM_EXP_ROOT=$run SUBSCORE_PATH=$run CUDA_VISIBLE_DEVICES=$GPU
    # README "(i) [NAVSIM-v1]" command; ray worker default threads_per_node = 8 (as shipped)
    "${pin[@]}" "$DATA_DIR/envs/drivor/bin/python" navsim/planning/script/run_pdm_score_multi_gpu.py \
      train_test_split=navtest agent=drivoR experiment_name=drivoR_nav1 \
      agent.checkpoint_path="$DATA_DIR/models/drivor/drivor_Nav1_25epochs.pth" \
      agent.config.image_backbone.model_weights="$DATA_DIR/models/drivor/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors" \
      agent.config.proposal_num=64 agent.config.refiner_ls_values=0.0 agent.config.image_backbone.focus_front_cam=false \
      agent.config.one_token_per_traj=true agent.config.refiner_num_heads=1 agent.config.tf_d_model=256 \
      agent.config.tf_d_ffn=1024 agent.config.area_pred=false agent.config.agent_pred=false agent.config.ref_num=4 \
      agent.config.noc=1 agent.config.dac=1 agent.config.ddc=0.0 agent.config.ttc=5 agent.config.ep=5 agent.config.comfort=2 \
      metric_cache_path="$CACHE/v1_navtest" output_dir="$run" "${@:2}" ;;
  wajepa)
    cd "$TP/wajepa"
    py=$DATA_DIR/envs/wajepa/bin/python
    ex=$DATA_DIR/jev-drive/scripts/top10_t2/wajepa_export.py
    tdir=$run/trajectory_cache
    traj=$tdir/navtest_trajectories.pkl
    # configs/wa_jepa_hugsim.yaml = the EPDMS preset with only the V-JEPA 2.1 pretrained-encoder load disabled
    # (the full state dict is restored strictly afterwards) plus an unused hugsim: block; see the repo's own diff note.
    # Export = the repo's _run_worker made resumable (wajepa_export.py); NPROC ranks share one GPU.
    export PYTHONPATH=$TP/wajepa:$TP/navsim
    mkdir -p "$tdir"
    if [[ ! -f $traj ]]; then
      pids=()
      for r in $(seq 0 $((NPROC - 1))); do
        CUDA_VISIBLE_DEVICES=$GPU "${pin[@]}" "$py" "$ex" --rank "$r" --num-shards "$NPROC" --out-dir "$tdir" \
          --config configs/wa_jepa_hugsim.yaml --checkpoint "$DATA_DIR/models/wajepa/model_state_dict.pt" \
          --navsim-root "$TP/navsim" --openscene-root "$OPENSCENE_DATA_ROOT" > "$tdir/rank_$r.log" 2>&1 &
        pids+=($!)
      done
      tail -qF "$tdir"/rank_*.log | grep --line-buffered "\[export\]" &
      tailer=$!
      rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done; kill $tailer
      (( rc == 0 )) || { echo "an export rank failed"; exit 1; }
      "$py" "$ex" --merge --num-shards "$NPROC" --out-dir "$tdir" --navsim-root "$TP/navsim"
    fi
    echo "[$(date +%T)] export done; scoring v2 (EPDMS)"
    "${pin[@]}" "$py" -m eval.navsim_score_trajectory_cache --navsim-version v2 \
      --navsim-root "$TP/navsim" --openscene-root "$OPENSCENE_DATA_ROOT" --metric-cache-path "$CACHE/v2_navtest" \
      --trajectory-cache-path "$traj" --output-dir "$run/v2" --experiment-name worldmodel_navtest_v2_cached \
      --override worker.max_workers=$SCORE_WORKERS
    echo "[$(date +%T)] scoring v1.1 (PDMS)"
    PYTHONPATH=$TP/wajepa:$TP/navsim-v1.1 "${pin[@]}" "$py" -m eval.navsim_score_trajectory_cache --navsim-version v1 \
      --navsim-root "$TP/navsim-v1.1" --openscene-root "$OPENSCENE_DATA_ROOT" --metric-cache-path "$CACHE/v1_navtest" \
      --trajectory-cache-path "$traj" --output-dir "$run/v1_1" --experiment-name worldmodel_navtest_v1_1_cached \
      --v1-sensor-path-key jev_unused_sensor_path --override worker=single_machine_thread_pool \
      --override worker.max_workers=$SCORE_WORKERS || echo "v1.1 scoring failed" ;;
  *) echo "unknown model $model" >&2; exit 1 ;;
esac
echo "[$(date +%T)] done -> $run"
