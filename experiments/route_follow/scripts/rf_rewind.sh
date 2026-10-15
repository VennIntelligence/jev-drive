#!/usr/bin/env bash
# RECOVERY-EXPERT REWIND on the GPU box (prereg experiments/route_follow/plans/2026-10-16-recovery-rewind-prereg.md; rf_rewind.py; driver
# experiments/onpol_pai/lib/onp_log_driver.py ONP_CELLS rewind cells). One scene = one pool job: phase A = two launches side by side, `id`
# (RSG-L18-s0 alone: the failure and the control) and `xp0` (the rule expert from the start: the reference); tfail from `id`; phase B = one
# launch per Delta in 1 / 2 / 3 s, the policy replayed to k_fail - 10 Delta and the expert from there (same session: the policy's own state);
# then extract (AlpaSim venv) and read. Every launch is one rollout (rf_p4.sh's `stack`, box branch; JEV_TRACK=0 = the path of the stored
# r281/rsg read and the only path the expert serves on, decision 309's same-path rule).
#   rf_rewind.sh job <root> <scene id | 8-char id>    one scene (resumes: finished launches are skipped)
#   rf_rewind.sh fanout <root> <N | all | s8,s8>        one pool job per scene of $LIST not yet submitted (staged: 1, ~10, all); `all` adds the
#                                                      collect job
#   rf_rewind.sh collect <root>                        read (all scenes) + table -> <root>/{read.json, rewind.md, rewind.csv}
# nuPlan train 64 (prereg plans/2026-10-16-rewind-train64-prereg.md; rf_rewind.py n-*; driver sh30_driver.py with JEV_EXPERT + JEV_EXPERT_FROM;
# the held-read serving: rm_chain.sh readlist, plain RSG-L18-s0, 18 decisions at 2 Hz):
#   rf_rewind.sh train64 <root> units                  the 64 turn scenes of the stored train-pool follow table -> <root>/units.txt, scenes.txt
#   rf_rewind.sh train64 <root> pack                   pool CPU job: expert packs of those scenes (rf_expert.py pack on the stored roll configs)
#   rf_rewind.sh train64 <root> cache | ref             Amendment 2: the token + 14 s trajdata cache of the 64 scenes; the units / k_ref / h
#                                                      from the stored 18-decision rollouts (-> <root>/tfail.json, from_D<d>.txt)
#   rf_rewind.sh train64 <root> stage <name> <list> [chunks]   one stage, every job queued at once, 30 decisions each: `id` (+ `idB` with SMOKE=1), `xp0`
#                                                      and D1 / D2 / D3 (GPU, <chunks> jobs each; the D launches run the id chunk's own scene list,
#                                                      scenes without h replay the identity) -> read + table (CPU, after all)
#                                                      -> <root>/<name>/rewind_train64.{md,csv,json}
#   rf_rewind.sh train64 <root> read <stage,stage..> <out dir>   the combined read of several stages (job body of a stage's read too)
#   job body: train64 <root> nsim <run dir> <list> id|xp0|<from file>
# Self rewind (prereg plans/2026-10-16-rewind-self-prereg.md; rf_rewind.py n-self-*; driver sh30_driver.py with JEV_PERTURB, perturb.py): the same
# serving, cache and packs as train64 (<T64> = $DATA_DIR/runs/route_follow/rewind_train64); the policy drives to h = k_ref - 2 Delta (Delta 3 / 5 / 8 s),
# its served plan is transformed (gain g, arc scale s) for N decisions, then its raw plan again; the expert from the same h for comparison.
#   rf_rewind.sh self <root> units                     pool CPU job: failed units (d316 A without short) + 10 controls -> <root>/self.json, scenes.txt
#   rf_rewind.sh self <root> stage <name> <list>       one stage, every job queued at once: one GPU job per launch (`id`, + `idB` with SMOKE=1,
#                                                      `xpD<d>`, `D<d>-g<g>s<s>`; each launch runs the scenes of its file), then read + check
#                                                      (+ table) after all; AFTER=<job ids> gates the whole stage; VARIANTS / NPERT override
#                                                      the variants (default g 0.7 / 0.85 / 1.15 / 1.3 x s 0.9 / 1.1, 2 decisions)
#   rf_rewind.sh self <root> read <stage,stage..> [out dir]   the combined read + table (job body of a stage's read too)
#   job body: self <root> nsim <run dir> <list> id | xp=<file> | pt=<file>
# Env: LIST (scene list, default results/pai281/strict36.txt), PACK (expert packs, $DATA_DIR/runs/route_follow/rewind/pack_pai: Tokyo's
#   pack_pai of the 36 scenes), SMOKE (1: phase B adds `idB`, a second identity session), PRIO (pool priority, 6: below the running lanes),
#   JEV_PY (python with numpy; jobs run from a worktree without .venv), RW_VRAM (pool declaration, 22; measured peak 19.3 GB = phase B's 3 launches), RW_HIST (history key, alpasim-p4c1). Expert variant XP-L12-E46 (rule_expert defaults: lookahead 1.2 s,
#   entry 4.6 m/s). Serving as r281/rsg: SH30_TAG=RSG-L18-s0 JEV_VCONT=1.0 JEV_LEAD=1, gains 20, send_recording_ground_truth=true.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); repo=$(cd "$here/../../.." && pwd)
cmd=${1:?job|fanout|collect}; O=$(mkdir -p "${2:?root}" && cd "$2" && pwd); shift 2
D=$DATA_DIR; PY=${JEV_PY:-$repo/.venv/bin/python}; APY=$D/third_party/alpasim/.venv/bin/python
PACK=${PACK:-$D/runs/route_follow/rewind/pack_pai}; LIST=${LIST:-$repo/experiments/route_follow/results/pai281/strict36.txt}
WIZ=(runtime.simulation_config.send_recording_ground_truth=true controller.gains.rel_front_steering_angle_weight=20)
py() { $PY "$here/rf_rewind.py" "$@"; }
apy() { "$APY" "$here/rf_rewind.py" "$@" 2> >(grep -v 'Creating\|it/s' >&2); }

stack() {  # <launch dir> <cells json> <scene list>: one rollout
  local r=$1 cj=$2 sl=$3 rc
  rm -rf "$r" "$r.out"; mkdir -p "$r/sim/driver-logs"; cp "$cj" "$r/sim/driver-logs/p4cells.json"; ln -sfn sim/driver-logs "$r/driver"
  env SH30_TAG=RSG-L18-s0 JEV_VCONT=1.0 JEV_LEAD=1 JEV_TRACK=0 DRV_PY=../../onpol_pai/lib/onp_log_driver.py CONC=1 \
    bash "$repo/experiments/alpasim/scripts/pai_native.sh" "$r/sim" "$sl" "${WIZ[@]}" runtime.simulation_config.n_rollouts=1 > "$r.out" 2>&1; rc=$?
  rm -rf "$r/sim/nre-cache" "$r/sim/driver-tmp"; find "$r/sim/rollouts" -name '*.mp4' -delete 2>/dev/null
  local got; got=$($PY -c "import json,sys;print(sum('error' not in (x.get('metrics') or {}) for x in json.load(open(sys.argv[1]))['rollouts']))" "$r/sim/aggregate/results-summary.json" 2>/dev/null || echo 0)
  echo "$(date '+%F %T') $(basename "$(dirname "$r")" | cut -c8-15) $(basename "$r"): rc $rc, $got / 1 rollouts" >> "$O/log.txt"
  (( rc == 0 && got == 1 )) && date > "$r/DONE"
}

phase() {  # <scene dir> <tags...>: the launches side by side (20 s apart: renderer start-up)
  local sd=$1 ok=1 pids=() tg; shift
  for tg in "$@"; do
    [[ -f $sd/$tg/DONE ]] && continue
    stack "$sd/$tg" "$sd/cells/cells_$tg.json" "$sd/scene.txt" & pids+=($!); sleep 20
  done
  for p in "${pids[@]}"; do wait "$p" || ok=0; done
  for tg in "$@"; do [[ -f $sd/$tg/DONE ]] || ok=0; done
  (( ok ))
}

job() {
  local sc=$1 sd tags
  [[ $sc == clipgt-* ]] || sc=$(grep -o "clipgt-$sc[0-9a-f-]*" "$LIST" | head -1)
  sd=$O/$sc; mkdir -p "$sd"; echo "$sc" > "$sd/scene.txt"; rm -f "$sd"/{DONE,ERROR}
  fail() { echo "$*" > "$sd/ERROR"; echo "$(date '+%F %T') ${sc:7:8} ERROR $*" >> "$O/log.txt"; return 1; }
  [[ -f $PACK/$sc.npz ]] || { fail "no expert pack $PACK/$sc.npz"; return 1; }
  tags=$(py cells --scene "$sc" --phase A --pack "$PACK" --out "$sd/cells") || { fail "cells A"; return 1; }
  phase "$sd" $tags || { fail "phase A (see $sd/*.out)"; return 1; }
  apy extract --sd "$sd" --out "$sd/a.pkl" > "$sd/extract_a.log" 2>&1 || { fail "extract A (extract_a.log)"; return 1; }
  py tfail --pkl "$sd/a.pkl" --out "$sd/tfail.json" > /dev/null || { fail "tfail"; return 1; }
  tags=$(py cells --scene "$sc" --phase B --pack "$PACK" --tfail "$sd/tfail.json" --out "$sd/cells" ${SMOKE:+--smoke}) || { fail "cells B"; return 1; }
  if [[ -n $tags ]]; then phase "$sd" $tags || { fail "phase B (see $sd/*.out)"; return 1; }; fi
  apy extract --sd "$sd" --out "$sd/all.pkl" > "$sd/extract.log" 2>&1 || { fail "extract (extract.log)"; return 1; }
  py read --root "$O" --scenes "${sc:7:8}" --out "$sd/read.json" > /dev/null || { fail "read"; return 1; }
  rm -f "$sd/a.pkl"; date > "$sd/DONE"
  echo "$(date '+%F %T') ${sc:7:8} done: $($PY -c "import json;t=json.load(open('$sd/tfail.json'));print(t['outcome'], t['src'], t['k_fail'], t['h'])")" >> "$O/log.txt"
}

train64() {
  local sub=${1:?units|pack|stage|nsim|read}; shift
  local P4=$D/runs/onpol_long/roll/RSG-L18-s0 NP=${RW_PACK:-$O/pack} W=$O/code AF=${RW_AFTER:-280}
  local CA=${RW_CACHE:-$D/datasets/alpasim_nuplan_long/after${RW_AFTER:-280}_rw}     # Amendment 2: 30 decisions = token + 14 s (num_timesteps_after counts 20 Hz steps)
  nsub() { local name=$1; shift; local id; id=$(cd "$repo" && .venv/bin/python -m jevdrive.cl submit --owner ${RW_OWNER:-rewind-train64} --name "$name" --tries ${TRIES:-2} \
           --priority ${PRIO:-6} --no-check "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }
           echo "$id $name" >> "$O/jobs.txt"; echo "$id"; }
  case $sub in
  units)
    $PY "$here/rf_rewind.py" n-units --follow $D/runs/route_follow/entry_speed/follow/scenes.csv --out "$O/units.txt" && cut -d' ' -f1 "$O/units.txt" > "$O/scenes.txt" ;;
  pack)
    nsub rwn-pack --vram 0.5 --cpu 16 --ram 64 --timeout-h 1 --log-dir "$O/pool/rwn-pack" -- \
      "$APY" "$repo/experiments/route_follow/scripts/rf_expert.py" pack --workers 16 --out "$NP" --scenes "$O/scenes.txt" --runs $(ls -d $P4/c*/run) ;;
  cache)  # pool CPU job: the trajdata cache at token + 14 s for the 64 scenes (alpasim_long/build_cache.py, as rf_lateturn_long.sh's after380_lt)
    nsub rwn-cache --tries 1 --vram 0.5 --cpu ${RW_CW:-16} --ram 64 --timeout-h 2 --log-dir "$O/pool/rwn-cache" -- env PYTHONPATH=$D/envs/nuplan_devkit_overlay \
      "$APY" "$repo/experiments/alpasim_long/scripts/build_cache.py" build --after $AF --out "$CA" --scenes "$O/scenes.txt" --workers ${RW_CW:-16} ;;
  ref)    # pool CPU job: units, strata, k_ref and h from the stored 18-decision rollouts -> <root>/tfail.json, from_D<d>.txt
    nsub rwn-ref --tries 1 --vram 0.5 --cpu 16 --ram 48 --timeout-h 1 --log-dir "$O/pool/rwn-ref" -- \
      "$APY" "$repo/experiments/route_follow/scripts/rf_rewind.py" n-tfail --stage "$O" --scenes "$O/scenes.txt" --runs "$(ls -d $P4/c*/run | paste -sd,)" ;;
  stage)
    local name=${1:?name} L=${2:?scene list} n=${3:-4} S i tg all=()
    [[ -s $O/tfail.json && -d $CA ]] || { echo "run train64 cache and ref first" >&2; exit 1; }
    S=$O/$name; [[ ! -e $S/SUBMITTED ]] || { echo "$S already submitted" >&2; exit 1; }
    [[ -d $W ]] || git -C "$repo" worktree add -q --detach "$W" HEAD || exit 1
    git -C "$W" checkout -q --detach "$(git -C "$repo" rev-parse HEAD)" || exit 1     # every stage serves the repo's HEAD (code.json per stage)
    echo "{\"served\": \"$W\", \"commit\": \"$(git -C "$W" rev-parse HEAD)\", \"dirty\": $(git -C "$W" status --porcelain -uno | grep -c .)}" > "$O/code.json"
    mkdir -p "$S/lists"; cp "$O/code.json" "$S/code.json"; for i in $(seq 0 $((n - 1))); do awk -v i=$i -v n=$n '(NR - 1) % n == i {print $1}' "$L" > "$S/lists/c$i.txt"; done
    local X="bash $W/experiments/route_follow/scripts/rf_rewind.sh train64 $O"
    local G=(--vram ${RW_VRAM:-20} --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=${RW_HIST:-alpasim-read4} --env "JEV_PY=$PY")
    for tg in id xp0 ${SMOKE:+idB}; do for i in $(seq 0 $((n - 1))); do
      id=$(nsub rwn-$tg-c$i "${G[@]}" --log-dir "$S/pool/$tg-c$i" -- $X nsim "$S/$tg/c$i/run" "$S/lists/c$i.txt" ${tg/idB/id}) || exit 1
      all+=("$id")
    done; done
    for d in 1 2 3; do for i in $(seq 0 $((n - 1))); do
      all+=($(nsub rwn-D$d-c$i "${G[@]}" --log-dir "$S/pool/D$d-c$i" -- $X nsim "$S/D$d/c$i/run" "$S/lists/c$i.txt" "$O/from_D$d.txt")) || exit 1
    done; done
    nsub rwn-read --after "$(IFS=,; echo "${all[*]}")" --env "JEV_PY=$PY" --vram 0.5 --cpu 16 --ram 64 --timeout-h 1 --log-dir "$S/pool/read" -- $X read "$S" > /dev/null || exit 1
    date > "$S/SUBMITTED"; tail -$(( ${#all[@]} + 2 )) "$O/jobs.txt" ;;
  scenes-all)  # rewind_onpol_all: every scene of the stored onpol-train rollouts -> <root>/scenes.txt
    $PY -c "import json,sys;print('\n'.join(sorted(r['clipgt_id'] for f in sys.argv[1:] for r in json.load(open(f))['rollouts'])))" \
      $P4/c*/run/aggregate/results-summary.json > "$O/scenes.txt" && wc -l < "$O/scenes.txt" ;;
  ostage)  # rewind_onpol_all (plans/2026-10-16-rewind-onpol-all-prereg.md): <name> <list> <id chunks> <D chunks>, all queued at once:
           # id (+ idB on chunk 0 with SMOKE=1) -> tfail (units from the stage's own identity, RF_K_END 29) + pack (from the id runs' after280
           # configs) -> D2 / D3 (the failed scenes only, <D chunks> each) -> read + yield; AFTER=<ids> gates the stage
    local name=${1:?name} L=${2:?scene list} n=${3:-4} nd=${4:-4} S i d id ids=() all=() aft=() rr=()
    [[ -d $CA || -n ${AFTER:-} ]] || { echo "no cache $CA (run cache first)" >&2; exit 1; }
    S=$O/$name; [[ ! -e $S/SUBMITTED ]] || { echo "$S already submitted" >&2; exit 1; }
    [[ -d $W ]] || git -C "$repo" worktree add -q --detach "$W" HEAD || exit 1
    git -C "$W" checkout -q --detach "$(git -C "$repo" rev-parse HEAD)" || exit 1
    echo "{\"served\": \"$W\", \"commit\": \"$(git -C "$W" rev-parse HEAD)\", \"dirty\": $(git -C "$W" status --porcelain -uno | grep -c .)}" > "$O/code.json"
    mkdir -p "$S/lists"; cp "$O/code.json" "$S/code.json"; cp "$L" "$S/list.txt"
    for i in $(seq 0 $((n - 1))); do awk -v i=$i -v n=$n '(NR - 1) % n == i {print $1}' "$L" > "$S/lists/c$i.txt"; rr+=("$S/id/c$i/run"); done
    [[ -n ${AFTER:-} ]] && aft=(--after "$AFTER")
    local X="bash $W/experiments/route_follow/scripts/rf_rewind.sh train64 $O" K=(--env RF_K_END=29 --env "RW_CACHE=$CA" --env "RW_PACK=$S/pack")
    local G=(--vram ${RW_VRAM:-20} --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=${RW_HIST:-alpasim-read4} --env "JEV_PY=$PY" "${K[@]}")
    for i in $(seq 0 $((n - 1))); do ids+=($(nsub rwo-id-c$i "${G[@]}" "${aft[@]}" --log-dir "$S/pool/id-c$i" -- $X nsim "$S/id/c$i/run" "$S/lists/c$i.txt" id)) || exit 1; done
    [[ -n ${SMOKE:-} ]] && { all+=($(nsub rwo-idB-c0 "${G[@]}" "${aft[@]}" --log-dir "$S/pool/idB-c0" -- $X nsim "$S/idB/c0/run" "$S/lists/c0.txt" id)) || exit 1; }
    local C=(--vram 0.5 --cpu 32 --ram 96 --timeout-h 2 --env "JEV_PY=$PY" "${K[@]}" --after "$(IFS=,; echo "${ids[*]}")")
    local tf pk
    tf=$(nsub rwo-tfail "${C[@]}" --log-dir "$S/pool/tfail" -- "$APY" "$W/experiments/route_follow/scripts/rf_rewind.py" n-tfail --stage "$S" \
         --scenes "$S/list.txt" --runs "$(IFS=,; echo "${rr[*]}")" --kend 29 --deltas 2,3 --rewind event,off --workers 32) || exit 1
    pk=$(nsub rwo-pack "${C[@]}" --log-dir "$S/pool/pack" -- "$APY" "$W/experiments/route_follow/scripts/rf_expert.py" pack --workers 32 \
         --out "$S/pack" --scenes "$S/list.txt" --runs "${rr[@]}") || exit 1
    all+=("${ids[@]}")
    for d in 2 3; do for i in $(seq 0 $((nd - 1))); do
      all+=($(nsub rwo-D$d-c$i "${G[@]}" --after "$tf,$pk" --log-dir "$S/pool/D$d-c$i" -- $X nsim "$S/D$d/c$i/run" "$S/from_D$d.txt@$i/$nd" "$S/from_D$d.txt")) || exit 1
    done; done
    nsub rwo-read --after "$(IFS=,; echo "${all[*]}")" --env "JEV_PY=$PY" --env RF_K_END=29 --vram 0.5 --cpu 32 --ram 96 --timeout-h 2 \
      --log-dir "$S/pool/read" -- $X oread "$S" > /dev/null || exit 1
    date > "$S/SUBMITTED"; tail -$(( ${#all[@]} + 3 )) "$O/jobs.txt" ;;
  oread)  # <stage dir>[,<stage dir>..] [out dir, default the first stage]: n-read (units = each stage's tfail.json) + n-yield
    local S=${2:-${1%%,*}} refs; mkdir -p "$S"
    refs=$(for s in ${1//,/ }; do printf '%s,' "$s/tfail.json"; done); refs=${refs%,}
    RF_K_END=29 "$APY" "$here/rf_rewind.py" n-read --stage "$1" --ref "$refs" --out "$S/read.json" --pkl "$S/pkl" --workers 32 || exit 1
    $PY "$here/rf_rewind.py" n-yield --read "$S/read.json" --md "$S/rewind_onpol_all.md" --csv "$S/rewind_onpol_all.csv" --units "$S/units.csv" \
      --rows "$S/rows.csv" --json "$S/summary.json" --d316 "$D/runs/route_follow/rewind_train64/tfail.json" > /dev/null || exit 1
    if [[ -n ${RW_ROWS_DIR:-} ]]; then    # the trainer-facing row list (recovery_envelope / rf_recov.sh)
      mkdir -p "$RW_ROWS_DIR"; cp "$S/rows.csv" "$S/units.csv" "$S/summary.json" "$RW_ROWS_DIR/"
      cat > "$RW_ROWS_DIR/README.md" <<EOF
# rewind_onpol_all label rows (plans/2026-10-16-rewind-onpol-all-prereg.md; read $S)

rows.csv: one line per 2 Hz decision of a valid expert handover session (RSG-L18-s0 drove to h = k_ref - 2 Delta, the rule expert
XP-L12-E46 served from h in the same session), decisions k >= h before the session's first zero, on rf_follow's junction stretch
[entry - 20, exit + 10] m. Columns:
- scene, log: clip id and its nuPlan log; delta (s, 2 or 3), h (handover decision), k (decision index, 0.5 s apart; rf_follow's k)
- run, rollout_id: the expert session (AlpaSim run dir + rollout id; driver/drive.jsonl under run/ holds the served = expert plan at k)
- completed: the session completed the turn (rf_follow); zero_free: no zero-score event (at-fault, offroad, dist_to_gt >= 4 m) at any k >= h
- label: completed AND zero_free (the registered label set; everything else is listed for reference only)
- gate_pai / gate_nuplan: label AND the d293 shape gate after h with the PAI thresholds / the nuPlan human p95 (summary.json gate_*)
- within_tol: handover state within the identity's run-to-run spread (plan 0.15 m, track 5 cm, v 0.05 m/s)
units.csv: one line per strict turn of the identity (outcome, k_ref, h per Delta). summary.json: the read's counts and thresholds.
EOF
    fi
    echo "-> $S/rewind_onpol_all.md" ;;
  nsim)   # <run dir> <list | from file@i/n (the i-th of n chunks of its scenes)> id|xp0|<from file>: one 30-decision launch of the held-read
          # serving (rm_chain's sim, the long cache), resumable
    local R=$1 L=$2 M=$3 E=()
    if [[ $L == *@*/* ]]; then
      local f=${L%@*} ij=${L##*@}; L=$(dirname "$R")/list.txt; mkdir -p "$(dirname "$R")"
      awk -v i=${ij%/*} -v n=${ij#*/} '(NR - 1) % n == i {print $1}' "$f" > "$L"
    fi
    if [[ ! -s $L ]]; then mkdir -p "$R"; echo "empty list" > "$R/EMPTY"; exit 0; fi
    [[ -f $R/aggregate/results-summary.json ]] && { echo "$R done"; exit 0; }
    case $M in id) E=(-u JEV_EXPERT -u JEV_EXPERT_FROM) ;; xp0) E=(-u JEV_EXPERT_FROM JEV_EXPERT=$NP) ;; *) E=(JEV_EXPERT=$NP JEV_EXPERT_FROM=$M) ;; esac
    rm -rf "$R"; mkdir -p "$R"
    local B; B=$(cd "$repo" && python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print(" ".join(B.OVERRIDES))')
    (cd "$repo" && env "${E[@]}" SH30_TAG=RSG-L18-s0 SH30_DUMP=0 bash experiments/alpasim/scripts/run.sh "$R" sh30 --scene-list "$L" $B \
       runtime.scene_provider.trajdata.cache_location=$CA runtime.scene_provider.trajdata.dataset.extra_params.num_timesteps_after=$AF) || exit 1
    [[ -f $R/aggregate/results-summary.json ]] || { echo "no results-summary" >&2; exit 1; }
    find "$R/rollouts" -name rollout.asl -delete ;;
  read)   # <stage dir>[,<stage dir>..] [out dir, default the first stage]
    local S=${2:-${1%%,*}}; mkdir -p "$S"
    "$APY" "$here/rf_rewind.py" n-read --stage "$1" --ref "$O/tfail.json" --out "$S/read.json" --pkl "$S/pkl" && \
      $PY "$here/rf_rewind.py" n-table --read "$S/read.json" --md "$S/rewind_train64.md" --csv "$S/rewind_train64.csv" > /dev/null && echo "-> $S/rewind_train64.md" ;;
  *) sed -n '2,30p' "$0"; exit 2 ;;
  esac
}

self_() {
  local sub=${1:?units|stage|nsim|read}; shift
  local T64=$D/runs/route_follow/rewind_train64 P4=$D/runs/onpol_long/roll/RSG-L18-s0 W=$O/code AF=280
  local NP=$D/runs/route_follow/rewind_train64/pack CA=$D/datasets/alpasim_nuplan_long/after280_rw
  nsub() { local name=$1; shift; local id; id=$(cd "$repo" && .venv/bin/python -m jevdrive.cl submit --owner rewind-self --name "$name" --tries ${TRIES:-2} \
           --priority ${PRIO:-0} --no-check "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }
           echo "$id $name" >> "$O/jobs.txt"; echo "$id"; }
  case $sub in
  units)
    nsub rws-units --tries 1 --vram 0.5 --cpu 16 --ram 48 --timeout-h 1 --log-dir "$O/pool/rws-units" -- \
      "$APY" "$repo/experiments/route_follow/scripts/rf_rewind.py" n-self-units --out "$O" --scenes "$T64/scenes.txt" --tfail "$T64/tfail.json" \
      --runs "$(ls -d $P4/c*/run | paste -sd,)" ;;
  stage)
    local name=${1:?name} L=${2:?scene list} S tag mode f id all=() aft=()
    [[ -s $O/self.json && -d $CA ]] || { echo "run self units first (and train64 cache)" >&2; exit 1; }
    S=$O/$name; [[ ! -e $S/SUBMITTED ]] || { echo "$S already submitted" >&2; exit 1; }
    [[ -d $W ]] || git -C "$repo" worktree add -q --detach "$W" HEAD || exit 1
    git -C "$W" checkout -q --detach "$(git -C "$repo" rev-parse HEAD)" || exit 1
    echo "{\"served\": \"$W\", \"commit\": \"$(git -C "$W" rev-parse HEAD)\", \"dirty\": $(git -C "$W" status --porcelain -uno | grep -c .)}" > "$O/code.json"
    mkdir -p "$S"; cp "$O/code.json" "$S/code.json"; cp "$L" "$S/list.txt"
    $PY "$here/rf_rewind.py" n-self-files --units "$O/self.json" --list "$S/list.txt" --out "$S/files" ${VARIANTS:+--variants "$VARIANTS"} \
      --n ${NPERT:-2} ${SMOKE:+--smoke} > /dev/null || exit 1
    [[ -n ${AFTER:-} ]] && aft=(--after "$AFTER")
    local X="bash $W/experiments/route_follow/scripts/rf_rewind.sh self $O"
    local G=(--vram ${RW_VRAM:-20} --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=${RW_HIST:-alpasim-read4} --env "JEV_PY=$PY" "${aft[@]}")
    while read -r tag mode f; do
      [[ $mode == id ]] || cut -d' ' -f1 "$f" > "${f%.txt}.list"
      id=$(nsub "rws-$tag" "${G[@]}" --log-dir "$S/pool/$tag" -- $X nsim "$S/$tag/c0/run" "$([[ $mode == id ]] && echo "$f" || echo "${f%.txt}.list")" \
           "$([[ $mode == id ]] && echo id || echo "$mode=$f")") || exit 1
      all+=("$id")
    done < "$S/files/launches.txt"
    nsub rws-read --after "$(IFS=,; echo "${all[*]}")" --env "JEV_PY=$PY" --env "SMOKE=${SMOKE:-}" --vram 0.5 --cpu 16 --ram 64 --timeout-h 1 \
      --log-dir "$S/pool/read" -- $X read "$S" > /dev/null || exit 1
    date > "$S/SUBMITTED"; tail -$(( ${#all[@]} + 1 )) "$O/jobs.txt" ;;
  nsim)   # <run dir> <list> id | xp=<file> | pt=<file>: one 30-decision launch of the held-read serving, resumable
    local R=$1 L=$2 M=$3 E=()
    if [[ ! -s $L ]]; then mkdir -p "$R"; echo "empty list" > "$R/EMPTY"; exit 0; fi
    [[ -f $R/aggregate/results-summary.json ]] && { echo "$R done"; exit 0; }
    case $M in id) E=(-u JEV_EXPERT -u JEV_EXPERT_FROM -u JEV_PERTURB) ;; xp=*) E=(-u JEV_PERTURB JEV_EXPERT=$NP JEV_EXPERT_FROM=${M#xp=}) ;;
               pt=*) E=(-u JEV_EXPERT -u JEV_EXPERT_FROM JEV_PERTURB=${M#pt=}) ;; *) echo "mode $M" >&2; exit 2 ;; esac
    rm -rf "$R"; mkdir -p "$R"
    local B; B=$(cd "$repo" && python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print(" ".join(B.OVERRIDES))')
    (cd "$repo" && env "${E[@]}" SH30_TAG=RSG-L18-s0 SH30_DUMP=0 bash experiments/alpasim/scripts/run.sh "$R" sh30 --scene-list "$L" $B \
       runtime.scene_provider.trajdata.cache_location=$CA runtime.scene_provider.trajdata.dataset.extra_params.num_timesteps_after=$AF) || exit 1
    [[ -f $R/aggregate/results-summary.json ]] || { echo "no results-summary" >&2; exit 1; }
    find "$R/rollouts" -name rollout.asl -delete ;;
  read)   # <stage dir>[,<stage dir>..] [out dir, default the first stage]; the smoke (SMOKE=1) also reads d316's stored identities
    local S=${2:-${1%%,*}} st=(); mkdir -p "$S"
    [[ -n ${SMOKE:-} ]] && st=(--stored "$T64/pilot10,$T64/rest54")
    "$APY" "$here/rf_rewind.py" n-self-read --stage "$1" --units "$O/self.json" --out "$S/read.json" --pkl "$S/pkl" "${st[@]}" || exit 1
    $PY "$here/rf_rewind.py" n-self-table --read "$S/read.json" --md "$S/rewind_self.md" --csv "$S/rewind_self.csv" --d316 "$T64/all/read.json" > /dev/null || exit 1
    $PY "$here/rf_rewind.py" n-self-check --read "$S/read.json" --out "$S/check.txt" \
      $([[ $1 != *,* && -f ${1%%,*}/files/launches.txt ]] && echo "--launches ${1%%,*}/files/launches.txt") $([[ -z ${SMOKE:-} ]] && echo --stage) ;;
  *) sed -n '2,45p' "$0"; exit 2 ;;
  esac
}

case $cmd in
job) job "$1" ;;
train64) train64 "$@" ;;
self) self_ "$@" ;;
fanout)
  N=${1:?N | all | s8,s8}; cd "$repo" || exit 1
  W=$O/code
  if [[ ! -d $W ]]; then git -C "$repo" worktree add -q --detach "$W" HEAD || exit 1; fi
  echo "{\"served\": \"$W\", \"commit\": \"$(git -C "$W" rev-parse HEAD)\", \"dirty\": $(git -C "$W" status --porcelain -uno | grep -c .)}" > "$O/code.json"
  arms=$($PY - "$O" "$N" "$D" "$LIST" <<'PY'
import csv, os, sys
O, N, D, L = sys.argv[1:]
have, ok = set(os.listdir(f"{D}/datasets/nurec/all-usdzs")), set()
for f in ("sim_scenes.csv", "sim_scenes_2604.csv"):
    ok |= {r["scene_id"] for r in csv.DictReader(open(f"{D}/third_party/alpasim/data/scenes/{f}")) if r["uuid"] + ".usdz" in have}
todo = [x.split()[0] for x in open(L) if x.strip() and not os.path.exists(f"{O}/{x.split()[0]}/SUBMITTED")]
sc = [s for s in todo if s in ok]
print(f"{len(todo) - len(sc)} unsubmitted scenes not in the scene cache yet", file=sys.stderr)
pick = sc if N == "all" else [s for s in sc if s[7:15] in N.split(",")] if "," in N or not N.isdigit() else sc[: int(N)]
print(",".join(s[7:15] for s in pick))
PY
)
  [[ -n $arms ]] || { echo "nothing left to submit"; exit 0; }
  col=(); [[ $N == all ]] && col=(--collect "env JEV_PY=$PY bash $W/experiments/route_follow/scripts/rf_rewind.sh collect $O")
  .venv/bin/python -m jevdrive.cl fanout --name rw --arms "$arms" --vram ${RW_VRAM:-22} --cpu 4 --ram 32 --timeout-h 3 --tries 2 --priority ${PRIO:-6} \
    --env CL_HIST_KEY=${RW_HIST:-alpasim-p4c1} --env "JEV_PY=$PY" --env "PACK=$PACK" --env "LIST=$LIST" ${SMOKE:+--env SMOKE=1} \
    --log-dir "$O/pool/{arm}" "${col[@]}" -- bash "$W/experiments/route_follow/scripts/rf_rewind.sh" job "$O" "{arm}" | tee -a "$O/jobs.txt" || exit 1
  for a in ${arms//,/ }; do s=$(grep -o "clipgt-$a[0-9a-f-]*" "$LIST" | head -1); mkdir -p "$O/$s"; date > "$O/$s/SUBMITTED"; done ;;
collect)
  py read --root "$O" --out "$O/read.json" && py table --read "$O/read.json" --tokyo "$repo/experiments/route_follow/results/pai281/APP1X-L18-s0.turns.json" \
    --md "$O/rewind.md" --csv "$O/rewind.csv" > /dev/null && echo "-> $O/rewind.md" ;;
*) sed -n '2,17p' "$0"; exit 2 ;;
esac
