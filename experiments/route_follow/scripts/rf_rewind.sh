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

case $cmd in
job) job "$1" ;;
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
