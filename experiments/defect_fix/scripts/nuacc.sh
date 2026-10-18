#!/usr/bin/env bash
# NUACC: does the PAI serve recipe (JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7) help on the nuPlan track on top of JEV_HOLD=stop3?
# Arms: jtn = JTN-L18-s0, rsg = RSG-L18-s0 (like-for-like reference). Reads per arm: arena held 2 228 @18 chunks under stop3 + held 460 @18
# (6 chunks, no hold: the stored held references are native, guard RSG + 5 zeros). Box, repo root, every job through the GPU pool.
#   nuacc.sh submit        arm rsg (priority 14) then arm jtn (13), then one summary job after all reads
#   nuacc.sh aread <arm> <chunk>   job body: one arena chunk (arena_hold.sh one_run with the sh30hold driver, JEV_HOLD=stop3 + SERVE)
#   nuacc.sh sum           job body: arena reports of both arms + the arm pair + held 460 -> $O/nuacc/report, prints DONE
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/defect_fix; N=$O/nuacc; ME=experiments/defect_fix/scripts/nuacc.sh; PY=experiments/defect_fix/scripts/jt.py; NP=experiments/defect_fix/scripts/nuacc.py
TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
AH=$DATA_DIR/runs/alpasim_proxy/arena/hold; AC=$DATA_DIR/datasets/alpasim_nuplan_arena/v1; KEY=runtime.scene_provider.trajdata
SERVE=(JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7)
tag() { [[ $1 == jtn ]] && echo JTN-L18-s0 || echo RSG-L18-s0; }
mkdir -p $N/pool
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner defect-fix --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-13} \
     --no-check --log-dir $N/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $N/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
case ${1:-} in
submit)
  ids=(); nch=$(python3 -c "import json; print(len(json.load(open('$AH/meta.json'))['chunks']))")
  E=(); for kv in "${SERVE[@]}"; do E+=(--env "$kv"); done
  for arm in rsg jtn; do
    T=$(tag $arm); P=13; [[ $arm == rsg ]] && P=14
    for c in $(seq 0 $((nch - 1))); do
      ids+=($(PRIO=$P TRIES=2 sub nuacc-ar-$arm-c$c --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 "${E[@]}" -- bash $ME aread $arm $c)) || exit 1; done
    for i in 0 1 2 3 4 5; do
      ids+=($(PRIO=$P TRIES=2 sub nuacc-held-$arm-$i --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 "${E[@]}" \
          -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6 accgap)) || exit 1; done
  done
  s=$(PRIO=13 sub nuacc-sum --after $(csv ${ids[@]}) --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME sum) || exit 1
  echo "$(date '+%F %T') nuacc: ${#ids[@]} reads ${ids[0]} .. ${ids[-1]} summary $s" | tee -a $N/stages.txt ;;
aread)
  arm=$2; c=$3; T=$(tag $arm); D=$N/$arm/arena/$T@stop3/c$c/run
  [[ -f $D/aggregate/results-summary.json ]] && { echo "$D already read"; exit 0; }
  k=$(python3 -c "import json; print(next(x['shard'] for x in json.load(open('$AH/meta.json'))['chunks'] if x['chunk'] == $c))") || die "no chunk $c"
  source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
  rm -rf $D; mkdir -p $D; rc=0
  env JEV_HOLD=stop3 SH30_TAG=$T bash experiments/alpasim/scripts/run.sh $D sh30hold --scene-list $AH/lists/c$c.txt "${BASE[@]}" "$KEY.cache_location=$AC/s$k" || rc=$?
  if [[ $rc -ne 0 ]]; then
    grep -q "exit code -9" "$D/native_summary.json" 2>/dev/null && { echo "runtime worker killed (-9): exit 137 for the pool's kill retry" >&2; exit 137; }
    die "run.sh rc $rc ($D)"; fi
  python3 - "$D" <<'PY'
import glob, json, os, sys
D = sys.argv[1]; n = 0
for r in json.load(open(f"{D}/aggregate/results-summary.json"))["rollouts"]:
    if not r.get("failure_reason") and (r.get("score") or 0) > 0:
        for f in glob.glob(f"{D}/rollouts/{r['clipgt_id']}/*/rollout.asl"):
            os.remove(f); n += 1
print("pruned", n, "rollout.asl")
PY
  ;;
sum)
  R=$N/report; mkdir -p $R
  for arm in jtn rsg; do $AP $PY arena --tags $(tag $arm)@stop3 --reads $N/$arm/arena --out $R/$arm > $R/$arm.stdout 2>&1 || die "arena $arm"; done
  $AP $NP pair --jtn $R/jtn/rows.csv.gz --rsg $R/rsg/rows.csv.gz --out $R/pair.md || die "pair"
  $TR $NP held > $R/held.txt 2>&1 || die "held"
  echo "$(date '+%F %T') DONE nuacc report $R" | tee $N/DONE ;;
*) sed -n 2,9p $0; exit 2 ;;
esac
