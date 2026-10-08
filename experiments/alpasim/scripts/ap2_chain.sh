#!/usr/bin/env bash
# AP2 (plans/2026-10-08-alpasim-aligned-prereg.md): one self-advancing chain per stage, run in tmux jev via scripts/tmux_run.sh. Every job goes
# through the pool; benchmark scoring through jevdrive.bench score-poses. Rerunning resumes (finished pool jobs are not resubmitted).
#   ap2_chain.sh pilot            routes + cold / backwarp tokens (navtest, shards 2-4) -> pilot arms N0 / A / AR / AB -> navtest offline read
#   ap2_chain.sh full <arm args>  routes + cold tokens of the other 9 shards -> AP2-<name>-s0 (SH30 recipe) -> navtest offline read
#                                 e.g. full A            (4-way route command, rule zero)   |   full AR --route   |   full AB --cold backwarp
#   ap2_chain.sh prepfull         routes + cold / backwarp tokens of the other 9 shards (can run while the pilot is read)
#   ap2_chain.sh loop <tag>       AlpaSim closed loop with the AP2 driver, tapped: 1 scene -> 3 -> 48 (8 concurrent), driver sanity after
#                                 each, route rebuild check, frame / BEV figures, per-scene table next to SH30 and LTF
# State: $DATA_DIR/runs/alpasim/ap2/chain-<stage>/{STATUS, DONE, ERROR, log.txt, jobs.txt}; pool logs under .../ap2/pool/.
set -uo pipefail
cd "$(dirname "$0")/../../.."
STAGE=$1; shift
A=$DATA_DIR/runs/alpasim/ap2
D=$A/chain-$STAGE; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
PYA=$DATA_DIR/third_party/alpasim/.venv/bin/python
VPY=$PWD/.venv/bin/python
CL="$VPY -m jevdrive.cl"
S=experiments/alpasim/scripts
L=$A/pool
SCENES=$DATA_DIR/runs/alpasim/scenes_navtest_full_part001_48.txt
K=12
status() { echo "$(date '+%F %T') ap2 $STAGE: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return
        $CL queue 2>/dev/null | awk -v p="$ld/log.txt" '($2 == "queued" || $2 == "running" || $2 == "inbox") && index($0, p) {f = 1} END {exit !f}' && return   # still live from an earlier start
        rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner alpasim --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 20; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; return 0; }
route() { sub ap2-route $L/route-$1 --vram 0.5 --cpu 18 --ram 40 -- $PYA $S/ap2_route.py build --data $1 --k4 $2 --workers 18; }
prep() { local d=$1; shift; sub ap2-prep $L/prep-$d --vram 30 --cpu 20 --ram 60 -- $PY $S/ap2_prep.py --data $d --workers 18 "$@"; }   # encoder at batch 128: 29 GB measured
offline() {  # name ref models... : plans on navtest, devkit scores of the subset, report
  local name=$1 ref=$2; shift 2
  sub ap2-offline $L/offline-$name --vram 20 --cpu 8 --ram 60 -- $PY $S/ap2_offline.py plans --name $name --scenes $SCENES --models "$@"
  waitdirs $L/offline-$name
  local keys; keys=${KEYS:-$($VPY -c "import numpy as np, sys; print(' '.join(k for k in np.load(sys.argv[1]).files if k.endswith(('_nav', '_m4', '_m1'))))" $A/offline/$name/poses.npz)}
  [[ -f $A/offline/$name/scores.csv ]] || $VPY -m jevdrive.bench score-poses --poses $A/offline/$name/poses.npz --keys $keys --tokens $A/offline/$name/subset.txt \
      --out $A/offline/$name/scores.csv --traffic non_reactive --priority 5 --wait || die "score-poses $name"   # 3.2 core-s per token and key: KEYS=... narrows the scored keys
  $PY $S/ap2_offline.py report --name $name --ref $ref --scores $A/offline/$name/scores.csv || die "report $name"
}

if [[ $STAGE == pilot ]]; then
  PILOT="navtrain_full.s2of12 navtrain_full.s3of12 navtrain_full.s4of12"
  status "routes and tokens: navtest + pilot shards"
  route lb_navtest fixed; for d in $PILOT; do route $d hash; done
  prep lb_navtest --bw; for d in $PILOT; do prep $d --bw; done
  waitdirs $L/route-lb_navtest $L/prep-lb_navtest $(for d in $PILOT; do echo $L/route-$d $L/prep-$d; done)
  T="$PY $S/ap2_train.py --data $PILOT --split navsim/op-parity-s234 --steps 3000 --batch 64 --warmup 100 --eval-every 1000 --seed 0"
  status "pilot arms N0 / A / AR / AB"
  sub ap2-t-pilot $L/t-N0 --train --vram 24 --cpu 6 --ram 40 -- $T --tag AP2P-N0-s0 --std navsim
  sub ap2-t-pilot $L/t-A  --train --vram 24 --cpu 6 --ram 40 -- $T --tag AP2P-A-s0
  sub ap2-t-pilot $L/t-AR --train --vram 24 --cpu 6 --ram 40 -- $T --tag AP2P-AR-s0 --route
  sub ap2-t-pilot $L/t-AB --train --vram 24 --cpu 6 --ram 40 -- $T --tag AP2P-AB-s0 --cold backwarp
  waitdirs $L/t-N0 $L/t-A $L/t-AR $L/t-AB
  status "offline read on navtest"
  offline pilot SHP SH30=SH30-F-s0 SHP=SHP-F-s0 N0=AP2P-N0-s0 A=AP2P-A-s0 AR=AP2P-AR-s0 AB=AP2P-AB-s0
elif [[ $STAGE == prepfull ]]; then   # the other 9 shards ahead of the arm choice: routes, cold and backwarp tokens (either rule can train on them)
  REST=$(for i in 0 1 5 6 7 8 9 10 11; do echo -n "navtrain_full.s${i}of$K "; done)
  status "routes and tokens of 9 shards"
  for d in $REST; do route $d hash; prep $d --bw; done
  waitdirs $(for d in $REST; do echo $L/route-$d $L/prep-$d; done)
elif [[ $STAGE == full ]]; then
  NAME=$1; shift
  REST=$(for i in 0 1 5 6 7 8 9 10 11; do echo -n "navtrain_full.s${i}of$K "; done)
  ALL=$(for i in $(seq 0 $((K - 1))); do echo -n "navtrain_full.s${i}of$K "; done)
  BW=""; [[ " $* " == *" backwarp "* ]] && BW=--bw
  status "routes and tokens: shard 0 first"
  route navtrain_full.s0of$K hash; prep navtrain_full.s0of$K $BW
  waitdirs $L/route-navtrain_full.s0of$K $L/prep-navtrain_full.s0of$K
  status "routes and tokens: the other shards"
  for d in $REST; do route $d hash; prep $d $BW; done
  waitdirs $(for d in $REST; do echo $L/route-$d $L/prep-$d; done)
  T="$PY $S/ap2_train.py --data $ALL --split navsim/op-parity-full --batch 128 --warmup 300 --eval-every 1000 --seed 0"
  status "training AP2-$NAME-s0 (smoke first)"
  $CL submit --owner alpasim --name ap2-t-smoke --log-dir $L/t-smoke-$NAME --train --vram 40 --cpu 6 --ram 60 -- $T --steps 20 --eval-every 20 --tag smoke-ap2 "$@" > /dev/null || die "submit smoke"
  waitdirs $L/t-smoke-$NAME
  sub ap2-t-full $L/t-full-$NAME --train --vram 40 --cpu 8 --ram 60 -- $T --steps 10000 --tag AP2-$NAME-s0 "$@"
  waitdirs $L/t-full-$NAME
  status "offline read on navtest"
  offline full-$NAME SH30 SH30=SH30-F-s0 AP2=AP2-$NAME-s0 P=AP2P-$NAME-s0
elif [[ $STAGE == loop ]]; then
  TAG=$1; R=$DATA_DIR/runs/alpasim; TS=$(cat "$D/ts" 2>/dev/null || date +%Y%m%d-%H%M%S | tee "$D/ts")
  OV8="runtime.nr_workers=2 runtime.endpoints.renderer.n_concurrent_rollouts=8 runtime.endpoints.driver.n_concurrent_rollouts=8 runtime.endpoints.controller.n_concurrent_rollouts=8 defines.nre_cache_size=9"
  loop() {  # name scene-list dump overrides... : one tapped closed-loop run, then the driver-side sanity gate
    local n=$1 list=$2 dump=$3; shift 3; local o=$R/ap2_$n/$TS
    sub alpasim-ap2-$n $o/pool --vram 40 --cpu 16 -- env AP2_TAG=$TAG SH30_DUMP=$dump bash $S/run.sh $o ap2 --tap --scene-list $list +e2e_challenge_nuplan=full "$@"
    waitdirs $o/pool
    $VPY - "$o" <<'PYEOF' || die "driver sanity $n"
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1] + "/driver-logs/drive.jsonl")]
dr, cl = [r for r in rows if r["kind"] == "drive"], [r for r in rows if r["kind"] == "close"]
res = json.load(open(sys.argv[1] + "/aggregate/results-summary.json"))["rollouts"]
bad = [c for c in cl if c["drive"] != 10 or c["inference"] != 10 or c["inference_error"] or c["input_error"]]
keys = sorted({(r["k"], r["n_keys"], r["n_slots"]) for r in dr if r["k"] < 4})
print(f"sessions {len(cl)}, drive records {len(dr)}, scored rollouts {len(res)}, mean score {sum(r['score'] for r in res) / max(len(res), 1):.4f}, (k, keys, slots) {keys}, bad sessions {len(bad)}")
sys.exit(1 if bad or len(dr) != 10 * len(cl) or len(res) != len(cl) else 0)
PYEOF
  }
  status "closed loop: 1 scene"; loop s1 $R/scenes_sh30_1.txt 1
  status "closed loop: 3 scenes"; loop s3 $R/scenes_sh30_3.txt 3
  status "closed loop: 48 scenes"; loop full48 $SCENES 8 $OV8
  $PYA $S/ap2_route.py check --run $R/ap2_full48/$TS --out $A/route_check_ap2 > "$D/route_check.txt" 2>&1 || die "route check"
  for n in s3 full48; do $PY $S/sh30_report.py frames --run $R/ap2_$n/$TS --out $R/ap2_$n/$TS/$n --session ${SESSION:-2} || die "frames $n"; done
  $VPY $S/sh30_report.py table --runs ap2=$R/ap2_full48/$TS sh30=$(ls -d $R/sh30_full48_c8/20261008-125144) ltf=$(ls -d $R/ltf_full48_c8/20261008-121920) --out $R/ap2_full48/$TS/table.md > /dev/null || die "table"
else
  die "unknown stage $STAGE"
fi
status "done"; date > "$D/DONE"
