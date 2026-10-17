#!/usr/bin/env bash
# WAIT-SET (results/wait_set.md, scripts/wait_set.py): the local closed-loop set where the logged human waits at standstill, and its reads.
# Box, repo root. Every GPU job goes through the pool; stages are chained with --after.
#   wait_set.sh select                 CPU, ~1 min: windows from the raw logs (wait_set.py select) -> $W/select/<ts>/; then copy
#                                      set.json / set.txt / controls.txt / cands.csv to results/wait_set/ (the repo copy is the set)
#   wait_set.sh build [after]          pool CPU job: the cache $C from results/wait_set/set.json, the control check (the source scenes'
#                                      public windows against the shipped cache: build_cache.py compare, exact on every control) and
#                                      wait_set.py check
#   wait_set.sh read <tag> [after]     ONE COMMAND PER ARM: one pool job reads the whole set (variants a + b) with <tag> and writes the
#                                      outcome table $W/reads/<tag>/table/<tag>.csv. <tag>: a checkpoint in runs/op_parity/runs or P0
#                                      (sh30, the public 10-decision read's driver path: run.sh sh30 + c0b_chain.OVERRIDES, as ol_chain.sh
#                                      readlist; the only addition is the cache), starter (AlpaSim's go-straight starter kit) or replay
#                                      (human replay: the starter kit under AlpaSim's +exp/sim=force_gt preset: the ego follows the log).
#                                      Prints the job id. Env PRIO (default 9).
#   wait_set.sh chain                  the first read: build -> starter + replay -> gate -> P0, RSG-L18-s0, RTB1ARO-L18-s0 -> report
#   wait_set.sh gate                   job body: go-straight scores >= 0.99 on variant a and the replay >= 0.98 on a and b, else rc 1
#                                      (a failed gate fails every read chained after it)
#   wait_set.sh report [tag ...]       job body / by hand: tables of every read (or the tags) -> $W/report/{tables.md, all.csv}
#   wait_set.sh smoke <tag> [n]        the first n windows of each variant (default 2) -> $W/smoke/<tag> (pool job; plumbing check)
#   wait_set.sh run <tag> <dir>        job body (WS_LIST overrides the scene list)
set -uo pipefail
cd "$(dirname "$0")/../../.."
DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}
E=experiments/alpasim_proxy; S=$E/results/wait_set; W=$DATA_DIR/runs/alpasim_proxy/wait_set; C=$DATA_DIR/datasets/alpasim_nuplan_wait/v1
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; PY="$P $E/scripts/wait_set.py"; CK=$DATA_DIR/runs/op_parity/runs
KEY=runtime.scene_provider.trajdata; CACHE=("$KEY.cache_location=$C")
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
mkdir -p $W/pool
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner alpasim-proxy --name "$name" --tries 1 --priority ${PRIO:-9} \
    --no-check --log-dir $W/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name"; echo "$id $name" >> $W/jobs.txt; echo "$id"; }
RD=(--vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4)     # the public reads' declaration (rtb-wide, rf_allrows)
STAGE=${1:?stage}; shift
case $STAGE in
select)
  $PY select --workers ${WS_WORKERS:-24} || die "select rc $?"
  d=$(ls -d $DATA_DIR/runs/alpasim_proxy/wait_set/select/2* | tail -1); mkdir -p $S
  cp $d/set.json $d/set.txt $d/controls.txt $S/ && gzip -c $d/cands.csv > $S/cands.csv.gz && echo "set from $d -> $S" ;;
build)
  [[ -f $S/set.json ]] || die "no $S/set.json"
  sub ws-build ${1:+--after $1} --vram 0.5 --cpu 24 --ram 48 --timeout-h 1 -- bash -c "set -e; [[ -f $C/nuplan_test/scenes_list.dill ]] || \
    PYTHONPATH=$DATA_DIR/envs/nuplan_devkit_overlay $PY build --set $S/set.json --out $C --workers 24; \
    $P experiments/alpasim_long/scripts/build_cache.py compare --a $C --b $DATA_DIR/datasets/alpasim_nuplan --scenes $S/controls.txt --out $C/controls_vs_shipped.json \
      | python3 -c 'import json, sys; s = json.load(sys.stdin); print(s); sys.exit(0 if s[\"errors\"] == 0 and s[\"exact\"] == s[\"scenes\"] else 1)'; \
    $PY check --set $S/set.json --cache $C" ;;
read)
  T=${1:?tag}; A=${2:-}; D=$W/reads/$T
  [[ ! -e $D/run/aggregate/results-summary.json ]] || die "$D already read"
  [[ $T == P0 || $T == starter || $T == replay || -f $CK/$T/ckpt-final.pt ]] || die "no checkpoint $T"
  mkdir -p $D; sub ws-read-$T ${A:+--after $A} "${RD[@]}" -- bash $E/scripts/wait_set.sh run $T $D/run ;;
smoke)
  T=${1:?tag}; n=${2:-2}; D=$W/smoke/$T; rm -rf $D; mkdir -p $D
  python3 -c "import json, sys; w = json.load(open('$S/set.json'))['windows']; print('\n'.join([x['id'] for x in w if x['variant'] == 'a'][:$n] + [x['id'] for x in w if x['variant'] == 'b'][:$n]))" > $D/list.txt
  sub ws-smoke-$T "${RD[@]}" --env WS_LIST=$D/list.txt -- bash $E/scripts/wait_set.sh run $T $D/run ;;
run)
  T=$1; D=$2; [[ -f $C/check.json ]] || die "no cache check $C/check.json"; [[ ! -e $D/aggregate/results-summary.json ]] || die "$D already ran"
  rm -rf "$D"; mkdir -p "$D"; L=${WS_LIST:-$S/set.txt}      # WS_LIST: a sub-list (smoke)
  case $T in
    starter) bash experiments/alpasim/scripts/run.sh "$D" starter --scene-list $L "${BASE[@]}" "${CACHE[@]}" || die "run.sh rc $?" ;;
    replay)  bash experiments/alpasim/scripts/run.sh "$D" starter --scene-list $L "${BASE[@]}" "${CACHE[@]}" +exp/sim=force_gt || die "run.sh rc $?" ;;
    *)       env SH30_TAG=$T bash experiments/alpasim/scripts/run.sh "$D" sh30 --scene-list $L "${BASE[@]}" "${CACHE[@]}" || die "run.sh rc $?" ;;
  esac
  $PY collect --set $S/set.json --run $D --label $T --out $(dirname $D)/table || die "collect rc $?"
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
gate)
  python3 - $W/reads/starter/table/starter.csv $W/reads/replay/table/replay.csv $S/set.txt <<'PY' | tee $W/gate.txt; exit ${PIPESTATUS[0]}
import csv, sys
n = len(open(sys.argv[3]).read().split())
g, r = (list(csv.DictReader(open(f))) for f in sys.argv[1:3])
mean = lambda rows, v: sum(float(x["score"]) for x in rows if x["variant"] == v) / max(1, sum(x["variant"] == v for x in rows))
ga, ra, rb = mean(g, "a"), mean(r, "a"), mean(r, "b")
bad = [(x["id"], x["score"], x["zero"]) for x in g if x["variant"] == "a" and float(x["score"]) < 1]
ok = len(g) == n and len(r) == n and ga >= 0.99 and ra >= 0.98 and rb >= 0.98
print(f"gate {'PASS' if ok else 'FAIL'}: windows {n}, go-straight rollouts {len(g)}, replay {len(r)}; go-straight a {ga:.4f} "
      f"(below 1: {bad}); replay a {ra:.4f}, b {rb:.4f}")
sys.exit(0 if ok else 1)
PY
  ;;
chain)
  b=$(bash $0 build) || exit 1
  s=$(bash $0 read starter $b) || exit 1; r=$(bash $0 read replay $b) || exit 1
  g=$(sub ws-gate --after $s,$r --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $E/scripts/wait_set.sh gate) || exit 1
  ids=(); for T in P0 RSG-L18-s0 RTB1ARO-L18-s0; do ids+=($(bash $0 read $T $g)) || exit 1; done
  q=$(sub ws-report --after $(IFS=,; echo "${ids[*]}") --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $E/scripts/wait_set.sh report) || exit 1
  echo "chain: build $b; starter $s replay $r; gate $g; reads ${ids[*]}; report $q" | tee -a $W/stages.txt ;;
report)
  mkdir -p $W/report; ins=()
  for T in ${@:-$(ls $W/reads)}; do [[ -d $W/reads/$T/table ]] && ins+=($W/reads/$T/table); done
  $PY report --in "${ins[@]}" --out $W/report/tables.md --csv $W/report/all.csv \
    --pairs RTB1ARO-L18-s0:RSG-L18-s0 P0:RSG-L18-s0 || die "report rc $?" ;;
*) die "unknown stage $STAGE" ;;
esac
