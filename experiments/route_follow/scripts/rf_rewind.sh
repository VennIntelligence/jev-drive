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
#   rf_rewind.sh train64 <root> stage <name> <list> [chunks]   one stage, every job queued at once: `id` (+ `idB` with SMOKE=1) and `xp0`
#                                                      (GPU, <chunks> jobs each) -> n-tfail (CPU, after id) -> D1 / D2 / D3 (GPU, after n-tfail)
#                                                      -> read + table (CPU, after all) -> <root>/<name>/rewind_train64.{md,csv,json}
#   rf_rewind.sh train64 <root> read <stage,stage..> <out dir>   the combined read of several stages (job body of a stage's read too)
#   job body: train64 <root> nsim <run dir> <list> id|xp0|<from file>
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
  local P4=$D/runs/onpol_long/roll/RSG-L18-s0 NP=$O/pack W=$O/code
  nsub() { local name=$1; shift; local id; id=$(cd "$repo" && .venv/bin/python -m jevdrive.cl submit --owner rewind-train64 --name "$name" --tries ${TRIES:-2} \
           --priority ${PRIO:-6} --no-check "$@" | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }
           echo "$id $name" >> "$O/jobs.txt"; echo "$id"; }
  case $sub in
  units)
    python3 "$here/rf_rewind.py" n-units --follow $D/runs/route_follow/entry_speed/follow/scenes.csv --out "$O/units.txt" && cut -d' ' -f1 "$O/units.txt" > "$O/scenes.txt" ;;
  pack)
    nsub rwn-pack --vram 0.5 --cpu 16 --ram 64 --timeout-h 1 --log-dir "$O/pool/rwn-pack" -- \
      "$APY" "$repo/experiments/route_follow/scripts/rf_expert.py" pack --workers 16 --out "$NP" --scenes "$O/scenes.txt" --runs $(ls -d $P4/c*/run) ;;
  stage)
    local name=${1:?name} L=${2:?scene list} n=${3:-4} S i tg ids=() tf all=()
    S=$O/$name; [[ ! -e $S/SUBMITTED ]] || { echo "$S already submitted" >&2; exit 1; }
    [[ -d $W ]] || git -C "$repo" worktree add -q --detach "$W" HEAD || exit 1
    echo "{\"served\": \"$W\", \"commit\": \"$(git -C "$W" rev-parse HEAD)\", \"dirty\": $(git -C "$W" status --porcelain -uno | grep -c .)}" > "$O/code.json"
    mkdir -p "$S/lists"; for i in $(seq 0 $((n - 1))); do awk -v i=$i -v n=$n '(NR - 1) % n == i {print $1}' "$L" > "$S/lists/c$i.txt"; done
    local X="bash $W/experiments/route_follow/scripts/rf_rewind.sh train64 $O"
    local G=(--vram ${RW_VRAM:-20} --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=${RW_HIST:-alpasim-read4} --env "JEV_PY=$PY")
    for tg in id xp0 ${SMOKE:+idB}; do for i in $(seq 0 $((n - 1))); do
      id=$(nsub rwn-$tg-c$i "${G[@]}" --log-dir "$S/pool/$tg-c$i" -- $X nsim "$S/$tg/c$i/run" "$S/lists/c$i.txt" ${tg/idB/id}) || exit 1
      all+=("$id"); [[ $tg == id ]] && ids+=("$id")
    done; done
    tf=$(nsub rwn-tfail --after "$(IFS=,; echo "${ids[*]}")" --vram 0.5 --cpu 16 --ram 48 --timeout-h 1 --log-dir "$S/pool/tfail" -- \
         "$APY" "$W/experiments/route_follow/scripts/rf_rewind.py" n-tfail --stage "$S" --chunks $n) || exit 1
    for d in 1 2 3; do for i in $(seq 0 $((n - 1))); do
      all+=($(nsub rwn-D$d-c$i --after "$tf" "${G[@]}" --log-dir "$S/pool/D$d-c$i" -- $X nsim "$S/D$d/c$i/run" "$S/list_D${d}_c$i.txt" "$S/from_D$d.txt")) || exit 1
    done; done
    nsub rwn-read --after "$(IFS=,; echo "${all[*]}")" --vram 0.5 --cpu 16 --ram 64 --timeout-h 1 --log-dir "$S/pool/read" -- $X read "$S" > /dev/null || exit 1
    date > "$S/SUBMITTED"; tail -$(( ${#all[@]} + 2 )) "$O/jobs.txt" ;;
  nsim)   # <run dir> <list> id|xp0|<from file>: one 18-decision launch of the held-read serving, resumable
    local R=$1 L=$2 M=$3 E=()
    if [[ ! -s $L ]]; then mkdir -p "$R"; echo "empty list" > "$R/EMPTY"; exit 0; fi
    [[ -f $R/aggregate/results-summary.json ]] && { echo "$R done"; exit 0; }
    case $M in id) E=(-u JEV_EXPERT -u JEV_EXPERT_FROM) ;; xp0) E=(-u JEV_EXPERT_FROM JEV_EXPERT=$NP) ;; *) E=(JEV_EXPERT=$NP JEV_EXPERT_FROM=$M) ;; esac
    rm -rf "$R"; mkdir -p "$(dirname "$R")"
    env "${E[@]}" bash "$repo/experiments/route_mem/scripts/rm_chain.sh" readlist RSG-L18-s0 "$L" "$R" ;;
  read)   # <stage dir>[,<stage dir>..] [out dir, default the first stage]
    local S=${2:-${1%%,*}}; mkdir -p "$S"
    "$APY" "$here/rf_rewind.py" n-read --stage "$1" --out "$S/read.json" --pkl "$S/pkl" && \
      $PY "$here/rf_rewind.py" n-table --read "$S/read.json" --md "$S/rewind_train64.md" --csv "$S/rewind_train64.csv" \
        --stored $D/runs/route_follow/entry_speed/follow/scenes.csv > /dev/null && echo "-> $S/rewind_train64.md" ;;
  *) sed -n '2,30p' "$0"; exit 2 ;;
  esac
}

case $cmd in
job) job "$1" ;;
train64) train64 "$@" ;;
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
