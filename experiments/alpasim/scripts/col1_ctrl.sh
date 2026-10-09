#!/usr/bin/env bash
# Lane COL1 controls: P2H10-F-s0 again on the three original chunk lists of the 700 public scenes (c0b/lists/chunk{0,1,2}.txt, the lists of
# the OT3 run: the simulator only reproduces for identical scene lists, decision 211), with every rollout.asl kept (run.sh directly, no
# pruning chain) and the uncompiled model passes of the original run (SH30_COMPILE=0). Three pool jobs, the third after the first.
#   scripts/tmux_run.sh col1-ctrl experiments/alpasim/scripts/col1_ctrl.sh          (box)
# State in $DATA_DIR/runs/alpasim/col1/ctrl/: STATUS, DONE | ERROR, log.txt, manifest.json {"P2H10-F-s0": [run dirs]}.
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/alpasim/col1/ctrl; L=$DATA_DIR/runs/alpasim/c0b/lists; mkdir -p "$O"; rm -f "$O/DONE" "$O/ERROR"
exec > >(tee -a "$O/log.txt") 2>&1
st() { echo "$(date '+%F %T') $*" | tee "$O/STATUS"; }
OV=(+e2e_challenge_nuplan=full runtime.nr_workers=2 runtime.endpoints.renderer.n_concurrent_rollouts=8
    runtime.endpoints.driver.n_concurrent_rollouts=8 runtime.endpoints.controller.n_concurrent_rollouts=8 defines.nre_cache_size=9)
ts=$(date +%Y%m%d-%H%M%S); dirs=(); first=
for c in 0 1 2; do
  D=$O/runs/chunk$c/$ts; mkdir -p "$D"; dirs+=("$D"); n=$(wc -l < "$L/chunk$c.txt")
  after=(); [[ $c == 2 ]] && after=(--after "$first")
  id=$(.venv/bin/python -m jevdrive.cl submit --owner alpasim-col1 --name "col1-ctrl-chunk$c" --vram 32 --cpu 8 --ram $((14 + n * 12 / 100)) \
       --timeout-h 3 --tries 1 --priority 12 --log-dir "$D/pool" "${after[@]}" -- env SH30_TAG=P2H10-F-s0 SH30_COMPILE=0 \
       bash experiments/alpasim/scripts/run.sh "$D" sh30 --scene-list "$L/chunk$c.txt" "${OV[@]}" | awk '{print $NF}') || { echo submit > "$O/ERROR"; exit 1; }
  [[ $c == 0 ]] && first=$id
  echo "chunk$c -> $id $D"
done
printf '{"P2H10-F-s0": ["%s", "%s", "%s"]}\n' "${dirs[@]}" > "$O/manifest.json"
t0=$SECONDS
while :; do
  ok=0
  for D in "${dirs[@]}"; do
    [[ -f $D/native_summary.json ]] && { grep -q '"rc": 0' "$D/native_summary.json" && ok=$((ok + 1)) || { st "ERROR $D rc != 0"; echo "$D" > "$O/ERROR"; exit 1; }; }
  done
  st "$ok / 3 chunks done, $(find "$O/runs" -name _complete | wc -l) rollouts, $(( (SECONDS - t0) / 60 )) min"
  (( ok == 3 )) && break
  (( SECONDS - t0 > 3 * 3600 )) && { st "ERROR timeout"; echo timeout > "$O/ERROR"; exit 1; }
  sleep 120
done
date > "$O/DONE"; st "done: $(find "$O/runs" -name rollout.asl | wc -l) rollout.asl, $(du -sh "$O/runs" | cut -f1)"
